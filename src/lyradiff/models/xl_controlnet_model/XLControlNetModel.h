#pragma once

#include "XLControlNetModelWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/controlnet_conditioning_embedding/ControlNetConditioningEmbedding.h"
#include "src/lyradiff/layers/controlnet_final_conv/ControlNetFinalConv.h"

#include "src/lyradiff/layers/cross_attn_downblock_2d/XLCrossAttnDownBlock2d.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/XLCrossAttnUpBlock2d.h"
#include "src/lyradiff/layers/down_block_2d/XLDownBlock2D.h"
#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlock.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/XLUNetMidBlock2DCrossAttn.h"
#include "src/lyradiff/layers/upblock2d/XLUpBlock2d.h"

#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class XLControlNetModel: public BaseLayer {
private:
    // params for 2 Identical ResNets
    std::vector<size_t> block_out_channels_ = {320, 640, 1280};
    std::vector<size_t> head_nums_          = {10, 20};
    std::vector<size_t> inner_trans_nums_   = {2, 10};
    bool                use_runtime_augemb_ = false;
    size_t              temb_channels_      = 1280;
    size_t              cross_attn_dim_     = 2048;
    size_t              norm_num_groups_    = 32;
    size_t              input_channels_     = 4;
    size_t              add_emb_dim_        = 256;

    size_t controlnet_condition_channels_ = 3;
    bool   is_reuse_unet_blocks_;

    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t hint_batch_size, size_t height, size_t width);

public:
    T* conv_buf_           = nullptr;
    T* conditioning_buf_   = nullptr;
    T* temb_buf_           = nullptr;
    T* time_proj_buf_      = nullptr;
    T* mid_hidden_res_buf_ = nullptr;

    std::vector<T*> down_hidden_states_bufs_ = std::vector<T*>(8, nullptr);
    // std::vector<std::vector<T*>> controlnet_res_bufs_ = std::vector<std::vector<T*>>(3, std::vector<T*>(13,
    // nullptr));
    std::vector<size_t> height_bufs_        = {0, 0, 0};
    std::vector<size_t> width_bufs_         = {0, 0, 0};
    size_t              cur_batch           = 0;
    size_t              cur_height          = 0;
    size_t              cur_width           = 0;
    size_t              cur_hint_batch_size = 0;

    TextTimeEmbeddingBlock<T>*    texttime_embedding         = nullptr;
    TimestepEmbeddingBlock<T>*    timestep_embedding         = nullptr;
    TimeProjection<T>*            time_proj                  = nullptr;
    XLDownBlock2D<T>*             down_block_2d              = nullptr;
    XLCrossAttnDownBlock2d<T>*    cross_attn_down_block_2d_1 = nullptr;
    XLCrossAttnDownBlock2d<T>*    cross_attn_down_block_2d_2 = nullptr;
    XLUNetMidBlock2DCrossAttn<T>* mid_block_2d               = nullptr;

    XLDownBlock2D<T>*             small_down_block_2d_1 = nullptr;
    XLDownBlock2D<T>*             small_down_block_2d_2 = nullptr;
    XLUNetMidBlock2DCrossAttn<T>* small_mid_block_2d    = nullptr;

    ControlNetConditioningEmbedding<T>* controlnet_conditioning_embedding = nullptr;
    ControlNetFinalConv<T>*             controlnet_final_conv             = nullptr;

    Conv2d<T>* input_conv_ = nullptr;

    XLControlNetModel(const bool          is_reuse_unet_blocks,
                      cudnnHandle_t       cudnn_handle,
                      cudaStream_t        stream,
                      cublasMMWrapper*    cublas_wrapper,
                      IAllocator*         allocator,
                      const bool          is_free_buffer_after_forward,
                      const bool          sparse,
                      const bool          use_runtime_augemb = false,
                      const LyraQuantType quant_level        = LyraQuantType::NONE);

    XLControlNetModel(XLControlNetModel<T> const& other);

    virtual ~XLControlNetModel();

    virtual void forward(std::vector<Tensor>&              output_tensors,
                         const TensorMap*                  input_tensors,
                         const float                       timestep,
                         const TensorMap*                  add_tensors,
                         const XLControlNetModelWeight<T>* weights,
                         const std::vector<float>&         controlnet_scales);
};
}  // namespace lyradiff
