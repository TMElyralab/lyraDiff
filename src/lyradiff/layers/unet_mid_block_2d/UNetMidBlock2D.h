#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d/UNetMidBlock2DWeight.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class UNetMidBlock2D: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t in_channels_;
    size_t temb_channels_;

    size_t ngroups_;
    bool   use_swish_;
    bool   use_flash_attn2_ = false;

    // params for Transformer2D
    size_t num_head_;
    size_t dim_per_head_;
    size_t encoder_hidden_dim_;
    T      qk_scale;
    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);
    void freeBuffer() override;

public:
    // blocks
    Resnet2DBlock<T>* resnet_ = nullptr;  // for resnet_0, resnet_1

    // buffer
    T* self_attn_qkv_buf_ = nullptr;
    T* self_attn_q_buf_   = nullptr;
    T* self_attn_k_buf_   = nullptr;
    T* self_attn_v_buf_   = nullptr;

    T* self_attn_qk_buf_ = nullptr;

    double* norm_cache_buf_ = nullptr;

    UNetMidBlock2D(const size_t     in_channels,
                   const size_t     temb_channels,
                   const size_t     ngroups,
                   const bool       use_swish,
                   const size_t     num_head,
                   cudnnHandle_t    cudnn_handle,
                   cudaStream_t     stream,
                   cublasMMWrapper* cublas_wrapper,
                   IAllocator*      allocator,
                   const bool       is_free_buffer_after_forward,
                   const bool       sparse);

    UNetMidBlock2D(UNetMidBlock2D<T> const& unet_mid_block_2d);

    virtual ~UNetMidBlock2D();

    virtual void forward(TensorMap*                     output_tensors,
                         const TensorMap*               input_tensors,
                         const UNetMidBlock2DWeight<T>* unet_mid_block_2d_weights);
};

}  // namespace lyradiff
