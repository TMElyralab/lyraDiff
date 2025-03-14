#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/context.h"
#include <iostream>

using namespace std;
namespace lyradiff {
template<typename T>
Conv2d<T>::Conv2d(const int                 in_channels,
                  const int                 out_channels,
                  const int                 kernel_size,
                  const int                 stride,
                  const int                 pad_h,
                  const int                 pad_w,
                  const cudnnTensorFormat_t input_format,
                  const cudnnTensorFormat_t output_format,
                  const cudnnTensorFormat_t kernel_format,
                  const cudnnTensorFormat_t bias_format,
                  cudaStream_t              stream,
                  cudnnHandle_t             cudnn_handle,
                  IAllocator*               allocator):
    in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    pad_h_(pad_h),
    pad_w_(pad_w),
    input_format_(input_format),
    output_format_(output_format),
    kernel_format_(kernel_format),
    bias_format_(bias_format),
    stream_(stream),
    cudnn_handle_(cudnn_handle),
    allocator_(allocator)
{
    if (std::is_same<T, half>()) {
        data_type_ = CUDNN_DATA_HALF;
    }
    else if (std::is_same<T, float>()) {
        data_type_ = CUDNN_DATA_FLOAT;
        // convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    }
    else {
        throw "Wrong Data type";
    }

    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_,
                                          /*dataType=*/data_type_,
                                          /*format=*/kernel_format_,
                                          /*out_channels=*/out_channels_,
                                          /*in_channels=*/in_channels_,
                                          /*kernel_height=*/kernel_size_,
                                          /*kernel_width=*/kernel_size_));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_,
                                               /*pad_height=*/pad_h_,
                                               /*pad_width=*/pad_w_,
                                               /*vertical_stride=*/stride_,
                                               /*horizontal_stride=*/stride_,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*//*CUDNN_CONVOLUTION,*/ CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/compute_type_));

    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor_,
                                          /*format=*/bias_format,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/1,
                                          /*channels=*/out_channels_,
                                          /*image_height=*/1,
                                          /*image_width=*/1));

    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor_));
    checkCUDNN(
        cudnnSetActivationDescriptor(activation_descriptor_, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0f));

    // checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor_, CUDNN_TENSOR_OP_MATH));

    cudnnSetStream(cudnn_handle_, stream_);
    workspace_size_ = 0;

    cur_w     = 0;
    cur_h     = 0;
    cur_batch = 0;
}

template<typename T>
Conv2d<T>::~Conv2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // if (workspace_size_ > 0) {
    //     allocator_->free((void**)(&workspace_));
    // }
    allocator_ = nullptr;

    checkCUDNN(cudnnDestroyTensorDescriptor(bias_descriptor_));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor_));
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor_));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor_));
}

template<typename T>
void Conv2d<T>::check_workspace(const cudnnTensorDescriptor_t input_descriptor_,
                                const cudnnTensorDescriptor_t output_descriptor_,
                                const int                     h,
                                const int                     w,
                                const int                     batch)
{
    if (cur_h != h || cur_w != w || cur_batch != batch) {
        cur_h     = h;
        cur_w     = w;
        cur_batch = batch;

        // int returned_algos;
        // checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle_,
        //                                                 input_descriptor_,
        //                                                 kernel_descriptor_,
        //                                                 convolution_descriptor_,
        //                                                 output_descriptor_,
        //                                                 9,
        //                                                 &returned_algos,
        //                                                 perf_results_));
        // printf("total %d algos.\n", returned_algos);
        // convolution_algorithm_ = perf_results_[0].algo;
        // float time             = perf_results_->time;
        // for (int i = 1; i < returned_algos; ++i) {
        //     printf("The algo: %d, cost time: %f\n", perf_results_[i].algo, perf_results_[i].time);
        //     if (perf_results_[i].status == CUDNN_STATUS_SUCCESS && perf_results_[i].time < time) {
        //         convolution_algorithm_ = perf_results_[i].algo;
        //         time                   = perf_results_->time;
        //     }
        // }
        // printf("best algos: %d, cost time: %f.\n", convolution_algorithm_, time);

        size_t cur_work_size;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_,
                                                           input_descriptor_,
                                                           kernel_descriptor_,
                                                           convolution_descriptor_,
                                                           output_descriptor_,
                                                           convolution_algorithm_,
                                                           &cur_work_size));
        // cur_work_size   = ((cur_work_size + 31) / 32) * 32;
        workspace_size_ = cur_work_size;
    }

    if (workspace_size_ == 0) {
        workspace_ = nullptr;
    }
    else {
        // cout << "cur conv workspace size " << workspace_size_ / 1024 / 1024 << "MBs" << endl;
        // workspace_ = (T*)allocator_->reMalloc(workspace_, workspace_size_, false);
        // cudaDeviceSynchronize();
        // if (std::is_same<T, float>()) {
        //     workspace_ = (T*)allocator_->reMallocWithName("Conv2d_workspace_", workspace_size_, false);
        // }
        // else {
        //     workspace_ = (T*)allocator_->reMalloc(workspace_, workspace_size_, false);
        // }

        workspace_ = (T*)allocator_->reMallocWithName("Conv2d_workspace_", workspace_size_, false);
    }
}

template<typename T>
void Conv2d<T>::conv2d(T*          output,
                       const T*    input,
                       const T*    kernel,
                       const int   batch,
                       const int   h,
                       const int   w,
                       const float alpha,
                       const float beta)
{
    // float alpha = 1.0f;
    // float beta  = 0.0f;

    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;
    // cudnnConvolutionFwdAlgoPerf_t
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/input_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/in_channels_,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_,
                                          /*format=*/output_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/out_channels_,
                                          /*image_height=*/(h + 2 * pad_h_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1,
                                          /*image_width=*/(w + 2 * pad_w_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1));

    check_workspace(input_descriptor_, output_descriptor_, h, w, batch);

    checkCUDNN(cudnnConvolutionForward(cudnn_handle_,
                                       &alpha,
                                       input_descriptor_,
                                       input,
                                       kernel_descriptor_,
                                       kernel,
                                       convolution_descriptor_,
                                       convolution_algorithm_,
                                       workspace_,
                                       workspace_size_,
                                       &beta,
                                       output_descriptor_,
                                       output));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));
}

template<typename T>
void Conv2d<T>::computeS3DiffLora(T* output, const T* input, const T* kernel, const int batch, const int h, const int w)
{
    // for nhwc: result = conv(x) + scale*convB(convA(x)w)
    if(weight_loader_manager_glob->ctx == nullptr) return;
    auto& lora_container =
        weight_loader_manager_glob->map_lora_container[weight_loader_manager_glob->ctx->cur_running_module];
    auto& unet_rev_weights =
        weight_loader_manager_glob->map_module_rev_weights[weight_loader_manager_glob->ctx->cur_running_module];
    auto weight_name = unet_rev_weights[(void*)kernel];
    PRINTF_S3DIFF("conv2dWithBias: conv weight name: %s, unet_rev_weights size: %d\n",
                  weight_name.c_str(),
                  unet_rev_weights.size());
    T* weight_alpha = (T*)lora_container.get_lora_weight(weight_name, true);
    // 不需要计算 lora 直接返回
    if (weight_alpha == nullptr)
        return;

    T* weight_beta = (T*)lora_container.get_lora_weight(weight_name, false);
    PRINTF_S3DIFF("get_de_mod: %s\n", weight_name.c_str());
    T* de_mod = (T*)lora_container.get_de_mod(weight_name);
    PRINTF_S3DIFF("get_scaling: %s\n", weight_name.c_str());
    float scaling = weight_loader_manager_glob->ctx->getParamVal("s3diff_lora_scaling");
    PRINTF_S3DIFF("scaling: %f\n", scaling);

    if (weight_alpha == nullptr || weight_beta == nullptr || de_mod == nullptr) {
        PRINTF_S3DIFF("conv hook get lora failed: %s, %d %d %d\n",
                      weight_name.c_str(),
                      weight_alpha == nullptr,
                      weight_beta == nullptr,
                      de_mod == nullptr);
        return;
    }

    PRINTF_S3DIFF("conv hook ok %s ,scaling: %f\n", weight_name.c_str(), scaling);
    int rank = int(8 / scaling);
    PRINTF_S3DIFF("rank: %d\n", rank);

    size_t workspace_size = sizeof(T) * batch * h * w * (max(rank, out_channels_) + rank) * 2;
    lora_workspace_       = (T*)allocator_->reMallocWithName("lora_workspace_", workspace_size, false);
    T* buffer1            = lora_workspace_;
    T* buffer2            = lora_workspace_ + batch * h * w * max(rank, out_channels_) * 2;

    // convA(x)
    Conv2d<T> conv_A = Conv2d<T>(in_channels_,
                                 rank,
                                 kernel_size_,
                                 stride_,
                                 pad_h_,
                                 pad_w_,
                                 input_format_,
                                 output_format_,
                                 kernel_format_,
                                 bias_format_,
                                 stream_,
                                 cudnn_handle_,
                                 allocator_);
    conv_A.conv2d(buffer1, input, weight_alpha, batch, h, w);
    PRINTF_S3DIFF("conv_A ok\n");
    // convA(x) @ de_mod
    int m = batch * h * w;
    int n = rank;
    int k = rank;
    PRINTF_S3DIFF("conv de_mod gemm mnk: [%d %d %d]\n", m, n, k);
    // printf("rank: %d\n", rank);
    cublas_wrapper_glob->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, de_mod, k, buffer1, k, buffer2, n);
    PRINTF_S3DIFF("conv de_mod gemm ok\n");

    // convB( (convA(x) @ de_mod) )
    Conv2d<T> conv_B = Conv2d<T>(rank,
                                 out_channels_,
                                 1,
                                 1,
                                 0,
                                 0,
                                 input_format_,
                                 output_format_,
                                 kernel_format_,
                                 bias_format_,
                                 stream_,
                                 cudnn_handle_,
                                 allocator_);
    conv_B.conv2d(buffer1, buffer2, weight_beta, batch, h, w);
    PRINTF_S3DIFF("conv_B gemm ok\n");
    invokeAddResidual(output, output, buffer1, 1, out_channels_, h, w, batch, stream_);
    PRINTF_S3DIFF("add residual ok\n");
}

template<typename T>
void Conv2d<T>::conv2dWithBias(T*          output,
                               const T*    input,
                               const T*    kernel,
                               const T*    bias,
                               const int   batch,
                               const int   h,
                               const int   w,
                               const float alpha,
                               const float beta)
{
    // float alpha = 1.0f;
    // float beta  = 0.0f;

    // cout << "cur conv params, in channel: " << in_channels_ << " out channel: " << out_channels_ << " kernel: " <<
    // kernel_size_ << " stride: " << stride_  << endl; cout << "cur conv input params, n: " << batch << " h: " << h <<
    // " w: " << w << " c: " << in_channels_ << endl; cout << endl;

    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/input_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/in_channels_,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_,
                                          /*format=*/output_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/out_channels_,
                                          /*image_height=*/(h + 2 * pad_h_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1,
                                          /*image_width=*/(w + 2 * pad_w_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1));

    check_workspace(input_descriptor_, output_descriptor_, h, w, batch);

    checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn_handle_,
                                                     &alpha,
                                                     input_descriptor_,
                                                     input,
                                                     kernel_descriptor_,
                                                     kernel,
                                                     convolution_descriptor_,
                                                     convolution_algorithm_,
                                                     workspace_,
                                                     workspace_size_,
                                                     &beta,
                                                     output_descriptor_,
                                                     output,
                                                     bias_descriptor_,
                                                     bias,
                                                     activation_descriptor_,
                                                     output_descriptor_,
                                                     output));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));

    this->computeS3DiffLora(output, input, kernel, batch, h, w);
}

template<typename T>
void Conv2d<T>::conv2dWithBiasWithResidual(T*          output,
                                           const T*    input,
                                           const T*    kernel,
                                           const T*    bias,
                                           const T*    residual,
                                           const int   batch,
                                           const int   h,
                                           const int   w,
                                           const float alpha,
                                           const float beta)
{
    // float alpha = 1.0f;
    // float beta  = 1.0f;

    // cout << "cur conv params, in channel: " << in_channels_ << " out channel: " << out_channels_ << " kernel: " <<
    // kernel_size_ << " stride: " << stride_  << endl; cout << "cur conv input params, n: " << batch << " h: " << h <<
    // " w: " << w << " c: " << in_channels_ << endl; cout << endl;

    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/input_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/in_channels_,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_,
                                          /*format=*/output_format_,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/out_channels_,
                                          /*image_height=*/(h + 2 * pad_h_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1,
                                          /*image_width=*/(w + 2 * pad_w_ - 1 * (kernel_size_ - 1) - 1) / stride_ + 1));

    check_workspace(input_descriptor_, output_descriptor_, h, w, batch);

    checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn_handle_,
                                                     &alpha,
                                                     input_descriptor_,
                                                     input,
                                                     kernel_descriptor_,
                                                     kernel,
                                                     convolution_descriptor_,
                                                     convolution_algorithm_,
                                                     workspace_,
                                                     workspace_size_,
                                                     &beta,
                                                     output_descriptor_,
                                                     residual,
                                                     bias_descriptor_,
                                                     bias,
                                                     activation_descriptor_,
                                                     output_descriptor_,
                                                     output));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));

    this->computeS3DiffLora(output, input, kernel, batch, h, w);
}

template<typename T>
void Conv2d<T>::transformTensor(T*                        output,
                                const T*                  input,
                                const cudnnTensorFormat_t input_format,
                                const cudnnTensorFormat_t output_format,
                                const int                 batch,
                                const int                 h,
                                const int                 w,
                                const int                 c,
                                const float               alpha,
                                const float               beta)
{
    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/input_format,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/c,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_,
                                          /*format=*/output_format,
                                          /*dataType=*/data_type_,
                                          /*batch_size=*/batch,
                                          /*channels=*/c,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(
        cudnnTransformTensor(cudnn_handle_, &alpha, input_descriptor_, input, &beta, output_descriptor_, output));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));
}

template class Conv2d<half>;
template class Conv2d<float>;
}  // namespace lyradiff