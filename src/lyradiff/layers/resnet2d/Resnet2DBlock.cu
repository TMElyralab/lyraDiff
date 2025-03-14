#include "Resnet2DBlock.h"
#include "src/lyradiff/kernels/activation_kernels.h"
using namespace std;
namespace lyradiff {

template<typename T>
Resnet2DBlock<T>::Resnet2DBlock(const size_t     in_channels,
                                const size_t     out_channels,
                                const size_t     ngroups_in,
                                const size_t     ngroups_out,
                                const bool       use_swish,
                                const size_t     time_emb_in_dim,
                                cudnnHandle_t    cudnn_handle,
                                cudaStream_t     stream_main,
                                cudaStream_t     stream_assistant,
                                cublasMMWrapper* cublas_wrapper,
                                IAllocator*      allocator,
                                bool             is_free_buffer_after_forward,
                                bool             has_temb):
    BaseLayer(stream_main, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    cudnn_handle_(cudnn_handle),
    stream_assistant_(stream_assistant),
    ngroups_in_(ngroups_in),
    ngroups_out_(ngroups_out),
    use_swish_(use_swish),
    time_emb_in_dim_(time_emb_in_dim),
    conv_shortcut_(false),
    has_temb_(has_temb)
{

    if (in_channels != out_channels) {
        conv_shortcut_ = true;
    }
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }

    // cout << "Resnet2DBlock in_channels " << in_channels << " out_channels " << out_channels << endl;

    input_conv_  = new Conv2d<T>(in_channels_,
                                out_channels_,
                                3,
                                1,
                                1,
                                1,
                                CUDNN_TENSOR_NHWC,
                                CUDNN_TENSOR_NHWC,
                                CUDNN_TENSOR_NHWC,
                                CUDNN_TENSOR_NHWC,
                                stream_main,
                                cudnn_handle,
                                allocator);
    second_conv_ = new Conv2d<T>(out_channels_,
                                 out_channels_,
                                 3,
                                 1,
                                 1,
                                 1,
                                 CUDNN_TENSOR_NHWC,
                                 CUDNN_TENSOR_NHWC,
                                 CUDNN_TENSOR_NHWC,
                                 CUDNN_TENSOR_NHWC,
                                 stream_main,
                                 cudnn_handle,
                                 allocator);
    if (conv_shortcut_) {
        shortcut_conv_ = new Conv2d<T>(in_channels_,
                                       out_channels_,
                                       1,
                                       1,
                                       0,
                                       0,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_TENSOR_NHWC,
                                       stream_main,
                                       cudnn_handle,
                                       allocator);
    }
}

template<typename T>
Resnet2DBlock<T>::Resnet2DBlock(const size_t     in_channels,
                                const size_t     out_channels,
                                const size_t     ngroups_in,
                                const size_t     ngroups_out,
                                const bool       use_swish,
                                cudnnHandle_t    cudnn_handle,
                                cudaStream_t     stream_main,
                                cudaStream_t     stream_assistant,
                                cublasMMWrapper* cublas_wrapper,
                                IAllocator*      allocator,
                                bool             is_free_buffer_after_forward,
                                bool             has_temb):
    Resnet2DBlock(in_channels,
                  out_channels,
                  ngroups_in,
                  ngroups_out,
                  use_swish,
                  1280,
                  cudnn_handle,
                  stream_main,
                  stream_assistant,
                  cublas_wrapper,
                  allocator,
                  is_free_buffer_after_forward)
{
}

template<typename T>
Resnet2DBlock<T>::Resnet2DBlock(Resnet2DBlock<T> const& resnet2DBlock):
    BaseLayer(resnet2DBlock.stream_,
              resnet2DBlock.cublas_wrapper_,
              resnet2DBlock.allocator_,
              resnet2DBlock.is_free_buffer_after_forward_,
              resnet2DBlock.cuda_device_prop_,
              resnet2DBlock.sparse_),
    in_channels_(resnet2DBlock.in_channels_),
    out_channels_(resnet2DBlock.out_channels_),
    cudnn_handle_(resnet2DBlock.cudnn_handle_),
    stream_assistant_(resnet2DBlock.stream_assistant_),
    ngroups_in_(resnet2DBlock.ngroups_in_),
    ngroups_out_(resnet2DBlock.ngroups_out_),
    use_swish_(resnet2DBlock.use_swish_),
    time_emb_in_dim_(resnet2DBlock.time_emb_in_dim_),
    conv_shortcut_(resnet2DBlock.conv_shortcut_),
    has_temb_(resnet2DBlock.has_temb_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }
    input_conv_    = resnet2DBlock.input_conv_;
    second_conv_   = resnet2DBlock.second_conv_;
    shortcut_conv_ = resnet2DBlock.shortcut_conv_;
}

template<typename T>
void Resnet2DBlock<T>::allocateBuffer()
{
}

template<typename T>
void Resnet2DBlock<T>::allocateBuffer(const size_t batch_size, const size_t input_bytes, const size_t output_bytes)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // size_t overall_size = 0;

    size_t temb_size1 = sizeof(T) * batch_size * time_emb_in_dim_;
    // size_t temb_size2 = sizeof(T) * batch_size * out_channels_;

    if (has_temb_) {
        temb_buf_1 = (T*)allocator_->reMallocWithName("Resnet2DBlock_temb_buf_1", temb_size1, false);
        // temb_buf_2 = (T*)allocator_->reMallocWithName("Resnet2DBlock_temb_buf_2", temb_size2, false);
        // overall_size += temb_size1;
    }

    inner_buf_1    = (T*)allocator_->reMallocWithName("Resnet2DBlock_inner_buf_1", input_bytes, false);
    inner_conv_buf = (T*)allocator_->reMallocWithName("Resnet2DBlock_inner_conv_buf", output_bytes, false);

    // conv_buf = (float*)allocator_->reMallocWithName("Resnet2DBlock_conv_buf", output_bytes * 4, false);

    gnorm1_caches_ = (double*)allocator_->reMallocWithName(
        "Resnet2DBlock_gnorm1_caches_", sizeof(double) * batch_size * ngroups_in_ * 2, false);
    gnorm2_caches_ = (double*)allocator_->reMallocWithName(
        "Resnet2DBlock_gnorm2_caches_", sizeof(double) * batch_size * ngroups_out_ * 2, false);

    // is_allocate_buffer_ = true;
}

template<typename T>
void Resnet2DBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        if (has_temb_) {
            allocator_->free((void**)(&temb_buf_1), false);
        }
        allocator_->free((void**)(&inner_buf_1), false);
        allocator_->free((void**)(&inner_conv_buf), false);
        allocator_->free((void**)(&gnorm1_caches_), false);
        allocator_->free((void**)(&gnorm2_caches_), false);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void Resnet2DBlock<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                               const std::vector<lyradiff::Tensor>* input_tensors,
                               const Resnet2DBlockWeight<T>*      weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}, {"temb", input_tensors->at(1)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void Resnet2DBlock<T>::forward(TensorMap*                    output_tensors,
                               const TensorMap*              input_tensors,
                               const Resnet2DBlockWeight<T>* weights)
{
    // input tensors:
    //      inpout_tensor:  [bs, height, width, in_channels],
    //      tem: [bs, 1280]

    // output tensors:
    //      output_tensor  [bs, height, width, out_channels],

    Tensor input_tensor = input_tensors->at("hidden_states");
    Tensor temb_tensor(MEMORY_GPU, TYPE_FP16, {0, 0}, nullptr);

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t height     = input_tensor.shape[1];
    size_t width      = input_tensor.shape[2];
    size_t nchannels  = input_tensor.shape[3];

    // 不要复用输入指针去做数值变动, 否则多轮计算后会修改输入 tensor 数值，定义为 const
    const T* input_buf    = input_tensor.getPtr<T>();
    T*       output_buf   = output_tensor.getPtr<T>();
    size_t   input_bytes  = sizeof(T) * input_tensor.size();
    size_t   output_bytes = sizeof(T) * output_tensor.size();

    allocateBuffer(batch_size, input_bytes, output_bytes);

    // 将 cublas wrapper 的流设置为助手流，并将其余 cublas_handle_ 绑定
    cublas_wrapper_->setStream(stream_assistant_);

    // 对 hidden_states 进行 GroupNorm 和 Silu 激活混合计算，使用主 stream_ 和 stream_assistant_ 并行计算
    invokeGroupNorm(inner_buf_1,
                    input_buf,
                    weights->gnorm1_gamma,
                    weights->gnorm1_beta,
                    gnorm1_caches_,
                    batch_size,
                    height,
                    width,
                    in_channels_,
                    ngroups_in_,
                    true,
                    stream_);

    // cout << "cur conv params, in channel: " << input_conv_->in_channels_ << " out channel: " <<
    // input_conv_->out_channels_ << " kernel: " << input_conv_->kernel_size_ << " stride: " << input_conv_->stride_  <<
    // endl; cout << "cur conv input params, n: " << batch_size << " h: " << height << " w: " << width << " c: " <<
    // input_conv_->in_channels_ << endl; cout << endl;

    input_conv_->conv2dWithBias(
        output_buf, inner_buf_1, weights->conv1_weight, weights->conv1_bias, batch_size, height, width);

    // // 处理 time embedding 线路，使用助手 流
    if (has_temb_) {
        Tensor   temb_tensor = input_tensors->at("temb");
        size_t   temb_dim    = temb_tensor.shape[1];
        const T* temb_buf    = temb_tensor.getPtr<T>();

        invokeGenericActivation<SiluActivation, T>(temb_buf_1, temb_buf, batch_size * temb_dim, stream_assistant_);

        // time embedding 的 Linear 层的 矩阵乘法， 助手流上操作
        // B: in: [Batch_size, D_in] ---> [Btch_size, temb_dim]
        // A: weight: [D_out, D_in] ----> [out_channels, tem_dim]
        // C: [Batch_size, D_out]  ----> [batch_size, out_channels]
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              out_channels_,                  // m
                              batch_size,                     // n
                              temb_dim,                       // k
                              weights->time_emb_proj_weight,  // A
                              temb_dim,                       // LDA
                              temb_buf_1,                     // B
                              temb_dim,                       // LDB
                              temb_buf_1,                     // C
                              out_channels_                   // LDC
        );

        invokeAddConvAndTemb(output_buf,
                             output_buf,
                             temb_buf_1,
                             weights->time_emb_proj_bias,
                             batch_size,
                             height,
                             width,
                             out_channels_,
                             stream_);
    }

    if (conv_shortcut_) {
        invokeGroupNorm(output_buf,
                        output_buf,
                        weights->gnorm2_gamma,
                        weights->gnorm2_beta,
                        gnorm2_caches_,
                        batch_size,
                        height,
                        width,
                        out_channels_,
                        ngroups_out_,
                        true,
                        stream_);

        second_conv_->conv2dWithBias(
            inner_conv_buf, output_buf, weights->conv2_weight, weights->conv2_bias, batch_size, height, width);

        shortcut_conv_->conv2dWithBiasWithResidual(output_buf,
                                                   input_buf,
                                                   weights->conv_shortcut_weight,
                                                   weights->conv_shortcut_bias,
                                                   inner_conv_buf,
                                                   batch_size,
                                                   height,
                                                   width,
                                                   1.0f,
                                                   1.0f);
    }
    else {
        invokeGroupNorm(inner_conv_buf,
                        output_buf,
                        weights->gnorm2_gamma,
                        weights->gnorm2_beta,
                        gnorm2_caches_,
                        batch_size,
                        height,
                        width,
                        out_channels_,
                        ngroups_out_,
                        true,
                        stream_);

        second_conv_->conv2dWithBiasWithResidual(output_buf,
                                                 inner_conv_buf,
                                                 weights->conv2_weight,
                                                 weights->conv2_bias,
                                                 input_buf,
                                                 batch_size,
                                                 height,
                                                 width,
                                                 1.0f,
                                                 1.0f);
    }
    // 还原外部传过来的 cublas_wraper 的 stream 到主流
    cublas_wrapper_->setStream(stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
Resnet2DBlock<T>::~Resnet2DBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete input_conv_;
    delete second_conv_;
    delete shortcut_conv_;
}

template class Resnet2DBlock<float>;
template class Resnet2DBlock<half>;

}  // namespace lyradiff