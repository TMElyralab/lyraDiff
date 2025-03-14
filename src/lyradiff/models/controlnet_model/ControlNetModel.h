#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/controlnet_conditioning_embedding/ControlNetConditioningEmbedding.h"
#include "src/lyradiff/layers/controlnet_final_conv/ControlNetFinalConv.h"
#include "src/lyradiff/layers/cross_attn_downblock_2d/CrossAttnDownBlock2d.h"
#include "src/lyradiff/layers/down_block_2d/DownBlock2D.h"
#include "src/lyradiff/layers/time_proj/TimeProjection.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlock.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/UNetMidBlock2DCrossAttn.h"
#include "src/lyradiff/models/controlnet_model/ControlNetModelWeight.h"

#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class ControlNetModel: public BaseLayer {
private:
    // params for 2 Identical ResNets
    std::vector<size_t> block_out_channels_            = {320, 640, 1280, 1280};
    size_t              temb_channels_                 = 1280;
    size_t              cross_attn_dim_                = 768;
    size_t              head_num_                      = 8;
    size_t              norm_num_groups_               = 32;
    size_t              input_channels_                = 4;
    size_t              controlnet_condition_channels_ = 3;
    bool                is_reuse_unet_blocks_;

    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t hint_batch_size, size_t height, size_t width);

public:
    T*              conv_buf_                = nullptr;
    T*              conditioning_buf_        = nullptr;
    T*              temb_buf_                = nullptr;
    T*              time_proj_buf_           = nullptr;
    std::vector<T*> down_hidden_states_bufs_ = std::vector<T*>(11, nullptr);
    T*              mid_hidden_res_buf_      = nullptr;

    std::vector<size_t> height_bufs_ = {0, 0, 0, 0};
    std::vector<size_t> width_bufs_  = {0, 0, 0, 0};
    size_t              cur_batch    = 0;

    TimestepEmbeddingBlock<T>*          timestep_embedding                = nullptr;
    TimeProjection<T>*                  time_proj                         = nullptr;
    CrossAttnDownBlock2d<T>*            cross_attn_down_block_2d_1        = nullptr;
    CrossAttnDownBlock2d<T>*            cross_attn_down_block_2d_2        = nullptr;
    CrossAttnDownBlock2d<T>*            cross_attn_down_block_2d_3        = nullptr;
    DownBlock2D<T>*                     down_block_2d                     = nullptr;
    UNetMidBlock2DCrossAttn<T>*         mid_block_2d                      = nullptr;
    ControlNetConditioningEmbedding<T>* controlnet_conditioning_embedding = nullptr;
    ControlNetFinalConv<T>*             controlnet_final_conv             = nullptr;

    Conv2d<T>* input_conv_ = nullptr;

    ControlNetModel(bool                is_reuse_unet_blocks,
                    cudnnHandle_t       cudnn_handle,
                    cudaStream_t        stream,
                    cublasMMWrapper*    cublas_wrapper,
                    IAllocator*         allocator,
                    const bool          is_free_buffer_after_forward,
                    const bool          sparse,
                    const LyraQuantType quant_level = LyraQuantType::NONE);

    ControlNetModel(ControlNetModel<T> const& unet);

    virtual ~ControlNetModel();

    virtual void forward(std::vector<Tensor>&            output_tensors,
                         const TensorMap*                input_tensors,
                         const float                     timestep,
                         const ControlNetModelWeight<T>* weights,
                         const std::vector<float>&       controlnet_scales);
};
}  // namespace lyradiff
