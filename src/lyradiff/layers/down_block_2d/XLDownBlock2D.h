#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/down_block_2d/XLDownBlock2DWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class XLDownBlock2D: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t in_channels_;
    size_t out_channels_;
    size_t temb_channels_;

    size_t ngroups_;
    bool   use_swish_;

    bool is_downsampler_;
    // handler
    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    void allocateBuffer() override;
    void freeBuffer() override;

public:
    // resnet2d blocks
    Resnet2DBlock<T>* resnet_0_         = nullptr;
    Resnet2DBlock<T>* resnet_1_         = nullptr;
    Conv2d<T>*        downsampler_conv_ = nullptr;

    XLDownBlock2D(const size_t     in_channels,
                  const size_t     out_channels,
                  const size_t     temb_channels,
                  const size_t     ngroups,
                  const bool       use_swish,
                  const bool       is_downsampler,
                  cudnnHandle_t    cudnn_handle,
                  cudaStream_t     stream,
                  cudaStream_t     stream_assistant,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  const bool       is_free_buffer_after_forward,
                  const bool       sparse);

    XLDownBlock2D(const size_t     in_channels,
                  const size_t     out_channels,
                  const size_t     temb_channels,
                  const size_t     ngroups,
                  const bool       is_downsampler,
                  cudnnHandle_t    cudnn_handle,
                  cudaStream_t     stream,
                  cudaStream_t     stream_assistant,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  const bool       is_free_buffer_after_forward,
                  const bool       sparse);

    XLDownBlock2D(const size_t     in_channels,
                  const size_t     out_channels,
                  const size_t     ngroups,
                  const bool       is_downsampler,
                  cudnnHandle_t    cudnn_handle,
                  cudaStream_t     stream,
                  cudaStream_t     stream_assistant,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  const bool       is_free_buffer_after_forward,
                  const bool       sparse);

    XLDownBlock2D(XLDownBlock2D<T> const& down_block_2d);

    virtual ~XLDownBlock2D();

    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const XLDownBlock2DWeight<T>* weights);
};

}  // namespace lyradiff
