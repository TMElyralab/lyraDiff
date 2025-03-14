#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlock.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T, bool ADD_UPSAMPLE>
class CrossAttnUpBlock2d: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        out_channels_;
    size_t        prev_output_channel_;
    size_t        temb_channels_;
    size_t        head_num_;
    size_t        norm_num_groups_;
    size_t        cross_attn_dim_;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width);

protected:
    Transformer2dBlock<T>* transformer2d_block_1;
    Transformer2dBlock<T>* transformer2d_block_2;
    Transformer2dBlock<T>* transformer2d_block_3;
    Resnet2DBlock<T>*      resnet2d_block1;
    Resnet2DBlock<T>*      resnet2d_block2;
    Resnet2DBlock<T>*      resnet2d_block3;
    Conv2d<T>*             upsampler_conv;

public:
    T* hidden_state_buf1_ = nullptr;
    T* hidden_state_buf2_ = nullptr;
    T* cat_buf1_          = nullptr;
    T* cat_buf2_          = nullptr;
    T* cat_buf3_          = nullptr;
    T* interpolate_buf_   = nullptr;

    CrossAttnUpBlock2d(const size_t        in_channels,
                       const size_t        out_channels,
                       const size_t        prev_output_channel,
                       const size_t        temb_channels,
                       const size_t        head_num,
                       const size_t        cross_attn_dim,
                       const size_t        norm_num_groups,
                       cudnnHandle_t       cudnn_handle,
                       cudaStream_t        stream,
                       cudaStream_t        stream_assistant,
                       cublasMMWrapper*    cublas_wrapper,
                       IAllocator*         allocator,
                       bool                is_free_buffer_after_forward,
                       const LyraQuantType quant_level = LyraQuantType::NONE);

    CrossAttnUpBlock2d(CrossAttnUpBlock2d<T, ADD_UPSAMPLE> const& cross_attn_down_block2d);

    virtual ~CrossAttnUpBlock2d();

    virtual void forward(std::vector<lyradiff::Tensor>*                     output_tensors,
                         const std::vector<lyradiff::Tensor>*               input_tensors,
                         const CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>* weights);
    virtual void forward(TensorMap*                                       output_tensors,
                         const TensorMap*                                 input_tensors,
                         const CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>* weights);
};

}  // namespace lyradiff
