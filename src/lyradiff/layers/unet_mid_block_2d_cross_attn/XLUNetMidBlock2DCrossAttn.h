#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/XLUNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class XLUNetMidBlock2DCrossAttn: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t in_channels_;
    size_t out_channels_;
    size_t temb_channels_;

    size_t ngroups_;
    bool   use_swish_;

    // params for Transformer2D
    size_t num_head_;
    size_t dim_per_head_;
    size_t encoder_hidden_dim_;
    size_t inner_trans_num_;

    // handler
    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);
    void freeBuffer() override;

public:
    // blocks
    Resnet2DBlock<T>*        resnet_ = nullptr;  // for resnet_0, resnet_1
    XLTransformer2dBlock<T>* attn_   = nullptr;

    // buffer
    T* inter_buf_ = nullptr;  // for resnet_0, attn

    XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                              const size_t     temb_channels,
                              const size_t     ngroups,
                              const bool       use_swish,
                              const size_t     num_head,
                              const size_t     encoder_hidden_dim,
                              const size_t     inner_trans_num,
                              cudnnHandle_t    cudnn_handle,
                              cudaStream_t     stream,
                              cudaStream_t     stream_assistant,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse,
                              LyraQuantType    quant_level = LyraQuantType::NONE);

    XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                              const size_t     temb_channels,
                              const size_t     ngroups,
                              const size_t     num_head,
                              const size_t     encoder_hidden_dim,
                              const size_t     inner_trans_num,
                              cudnnHandle_t    cudnn_handle,
                              cudaStream_t     stream,
                              cudaStream_t     stream_assistant,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse,
                              LyraQuantType    quant_level = LyraQuantType::NONE);

    XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                              const size_t     ngroups,
                              const size_t     num_head,
                              const size_t     encoder_hidden_dim,
                              cudnnHandle_t    cudnn_handle,
                              cudaStream_t     stream,
                              cudaStream_t     stream_assistant,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse,
                              LyraQuantType    quant_level = LyraQuantType::NONE);

    XLUNetMidBlock2DCrossAttn(XLUNetMidBlock2DCrossAttn<T> const& unet_mid_block_2d_cross_attn);

    virtual ~XLUNetMidBlock2DCrossAttn();

    virtual void forward(TensorMap*                                output_tensors,
                         const TensorMap*                          input_tensors,
                         const XLUNetMidBlock2DCrossAttnWeight<T>* unet_mid_block_2d_cross_attn_weights);
};

}  // namespace lyradiff
