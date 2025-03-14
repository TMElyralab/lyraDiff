#pragma once

#include "GLVUnet2dConditionalModelWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/cross_attn_downblock_2d/XLCrossAttnDownBlock2d.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/GLVCrossAttnUpBlock2d.h"
#include "src/lyradiff/layers/down_block_2d/XLDownBlock2D.h"
#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlock.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/XLUNetMidBlock2DCrossAttn.h"
#include "src/lyradiff/layers/upblock2d/GLVUpBlock2d.h"
#include "src/lyradiff/layers/zero_sft/ZeroSFT.h"
#include "src/lyradiff/models/xl_controlnet_model/GLVControlNetModel.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/context.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class GLVUnet2dConditionalModel: public BaseLayer {
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
    size_t              output_channels_    = 4;
    size_t              add_emb_dim_        = 256;
    // size_t              max_controlnet_num_;
    //  handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t controlnet_input_count);

public:
    T*              conv_buf_                = nullptr;
    T*              temb_buf_                = nullptr;
    T*              time_proj_buf_           = nullptr;
    std::vector<T*> down_hidden_states_bufs_ = std::vector<T*>(8, nullptr);
    std::vector<T*> up_hidden_states_bufs_   = std::vector<T*>(3, nullptr);
    std::vector<T*> controlnet_res_bufs_     = std::vector<T*>(10, nullptr);

    T*      mid_hidden_res_buf_   = nullptr;
    T*      mid_hidden_res_buf_2_ = nullptr;
    double* norm_cache_buf_       = nullptr;

    std::vector<size_t> height_bufs_ = {0, 0, 0};
    std::vector<size_t> width_bufs_  = {0, 0, 0};
    size_t              cur_batch    = 0;
    // size_t              cur_controlnet_count = 0;

    TextTimeEmbeddingBlock<T>*    texttime_embedding         = nullptr;
    TimestepEmbeddingBlock<T>*    timestep_embedding         = nullptr;
    TimeProjection<T>*            time_proj                  = nullptr;
    XLDownBlock2D<T>*             down_block_2d              = nullptr;
    XLCrossAttnDownBlock2d<T>*    cross_attn_down_block_2d_1 = nullptr;
    XLCrossAttnDownBlock2d<T>*    cross_attn_down_block_2d_2 = nullptr;
    XLUNetMidBlock2DCrossAttn<T>* mid_block_2d               = nullptr;
    GLVCrossAttnUpBlock2d<T>*     cross_attn_up_block_2d_1   = nullptr;
    GLVCrossAttnUpBlock2d<T>*     cross_attn_up_block_2d_2   = nullptr;
    GLVUpBlock2d<T>*              up_block_2d                = nullptr;

    ZeroSFT<T>* mid_block_project_module;

    Conv2d<T>* input_conv_  = nullptr;
    Conv2d<T>* output_conv_ = nullptr;

    GLVControlNetModel<T>* controlnet;

    GLVUnet2dConditionalModel(cudnnHandle_t    cudnn_handle,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse,
                              const bool       use_runtime_augemb = false,
                              const size_t     input_channels     = 4,
                              const size_t     output_channels    = 4);

    GLVUnet2dConditionalModel(GLVUnet2dConditionalModel<T> const& unet);

    virtual ~GLVUnet2dConditionalModel();

    virtual void forward(TensorMap*                                       output_tensors,
                         const TensorMap*                                 input_tensors,
                         const float                                      timestep,
                         const TensorMap*                                 add_tensors,
                         const GLVUnet2dConditionalModelWeight<T>*        unet_weights,
                         const std::vector<Tensor>*                       control_imgs,
                         const std::vector<Tensor>*                       controlnet_augs,
                         const std::vector<std::vector<float>>*           controlnet_scales,
                         const std::vector<GLVControlNetModelWeight<T>*>* controlnet_weights,
                         const bool                                       controlnet_guess_mode = false);
};
}  // namespace lyradiff
