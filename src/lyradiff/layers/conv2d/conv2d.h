#pragma once
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/cublasMMWrapper.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "stdlib.h"
#include <cuda_fp16.h>
#include <cudnn.h>

namespace lyradiff {

template<typename T>
class Conv2d {
private:
    T*     workspace_;
    T*     lora_workspace_;
    size_t workspace_size_;

    int pad_h_;
    int pad_w_;

    cudnnTensorFormat_t input_format_;
    cudnnTensorFormat_t output_format_;
    cudnnTensorFormat_t kernel_format_;
    cudnnTensorFormat_t bias_format_;

    cudnnFilterDescriptor_t      kernel_descriptor_;
    cudnnConvolutionDescriptor_t convolution_descriptor_;
    cudnnTensorDescriptor_t      bias_descriptor_;
    cudnnActivationDescriptor_t  activation_descriptor_;

    cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // cudnnConvolutionFwdAlgoPerf_t perf_results_[9];

    int cur_h;
    int cur_w;
    int cur_batch;

    cudaStream_t  stream_;
    cudnnHandle_t cudnn_handle_;

    cudnnDataType_t data_type_;
    cudnnDataType_t compute_type_ = CUDNN_DATA_FLOAT;
    // cudnnDataType_t compute_type_ = CUDNN_DATA_HALF;

    IAllocator* allocator_;

    virtual void check_workspace(const cudnnTensorDescriptor_t input_descriptor_,
                                 const cudnnTensorDescriptor_t output_descriptor_,
                                 const int                     h,
                                 const int                     w,
                                 const int                     batch);

protected:
    bool is_maintain_buffer = false;

public:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;

    Conv2d() = default;
    Conv2d(const int                 in_channels,
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
           IAllocator*               allocator);

    ~Conv2d();

    virtual void conv2d(T*          output,
                        const T*    input,
                        const T*    kernel,
                        const int   batch,
                        const int   h,
                        const int   w,
                        const float alpha = 1.0,
                        const float beta  = 0.0);

    virtual void conv2dWithBias(T*          output,
                                const T*    input,
                                const T*    kernel,
                                const T*    bias,
                                const int   batch,
                                const int   h,
                                const int   w,
                                const float alpha = 1.0,
                                const float beta  = 0.0);

    virtual void conv2dWithBiasWithResidual(T*          output,
                                            const T*    input,
                                            const T*    kernel,
                                            const T*    bias,
                                            const T*    residual,
                                            const int   batch,
                                            const int   h,
                                            const int   w,
                                            const float alpha = 1.0,
                                            const float beta  = 1.0);

    virtual void transformTensor(T*                        output,
                                 const T*                  input,
                                 const cudnnTensorFormat_t input_format,
                                 const cudnnTensorFormat_t output_format,
                                 const int                 batch,
                                 const int                 h,
                                 const int                 w,
                                 const int                 c,
                                 const float               alpha = 1.0,
                                 const float               beta  = 0.0);

    virtual void
    computeS3DiffLora(T* output, const T* input, const T* kernel, const int batch, const int h, const int w);
};  // namespace lyradiff

}  // namespace lyradiff
