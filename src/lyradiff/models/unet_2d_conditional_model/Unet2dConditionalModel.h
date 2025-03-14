#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/cross_attn_downblock_2d/CrossAttnDownBlock2d.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2d.h"
#include "src/lyradiff/layers/down_block_2d/DownBlock2D.h"
#include "src/lyradiff/layers/time_proj/TimeProjection.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlock.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlock.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/UNetMidBlock2DCrossAttn.h"
#include "src/lyradiff/layers/upblock2d/UpBlock2d.h"
#include "src/lyradiff/models/controlnet_model/ControlNetModel.h"
#include "src/lyradiff/models/controlnet_model/ControlNetModelWeight.h"
#include "src/lyradiff/models/unet_2d_conditional_model/Unet2dConditionalModel.h"
#include "src/lyradiff/models/unet_2d_conditional_model/Unet2dConditionalModelWeight.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class Unet2dConditionalModel: public BaseLayer {
private:
    // params for 2 Identical ResNets
    std::vector<size_t> block_out_channels_ = {320, 640, 1280, 1280};
    size_t              temb_channels_      = 1280;
    size_t              cross_attn_dim_     = 768;
    size_t              head_num_           = 8;
    size_t              norm_num_groups_    = 32;
    size_t              input_channels_     = 4;
    size_t              output_channels_    = 4;
    size_t              max_controlnet_num_;
    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t controlnet_input_count);

public:
    T*              conv_buf_                = nullptr;
    T*              temb_buf_                = nullptr;
    T*              time_proj_buf_           = nullptr;
    std::vector<T*> down_hidden_states_bufs_ = std::vector<T*>(11, nullptr);
    std::vector<T*> up_hidden_states_bufs_   = std::vector<T*>(4, nullptr);
    std::vector<T*> controlnet_res_bufs_     = std::vector<T*>(13, nullptr);

    T*      mid_hidden_res_buf_ = nullptr;
    double* norm_cache_buf_     = nullptr;

    std::vector<size_t> height_bufs_         = {0, 0, 0, 0};
    std::vector<size_t> width_bufs_          = {0, 0, 0, 0};
    size_t              cur_batch            = 0;
    size_t              cur_controlnet_count = 0;

    TimestepEmbeddingBlock<T>*    timestep_embedding         = nullptr;
    TimeProjection<T>*            time_proj                  = nullptr;
    CrossAttnDownBlock2d<T>*      cross_attn_down_block_2d_1 = nullptr;
    CrossAttnDownBlock2d<T>*      cross_attn_down_block_2d_2 = nullptr;
    CrossAttnDownBlock2d<T>*      cross_attn_down_block_2d_3 = nullptr;
    DownBlock2D<T>*               down_block_2d              = nullptr;
    UNetMidBlock2DCrossAttn<T>*   mid_block_2d               = nullptr;
    UpBlock2d<T>*                 up_block_2d                = nullptr;
    CrossAttnUpBlock2d<T, true>*  cross_attn_up_block_2d_1   = nullptr;
    CrossAttnUpBlock2d<T, true>*  cross_attn_up_block_2d_2   = nullptr;
    CrossAttnUpBlock2d<T, false>* cross_attn_up_block_2d_3   = nullptr;

    Conv2d<T>* input_conv_  = nullptr;
    Conv2d<T>* output_conv_ = nullptr;

    ControlNetModel<T>* controlnet;

    Unet2dConditionalModel(size_t max_controlnet_num,
                           //    std::vector<cudnnHandle_t> controlnet_cudnn_handles,
                           //    std::vector<cudaStream_t> controlnet_streams,
                           //    std::vector<cublasMMWrapper *> controlnet_cublas_wrappers,
                           //    std::vector<IAllocator *> controlnet_allocalors,
                           cudnnHandle_t      cudnn_handle,
                           cudaStream_t       stream,
                           cublasMMWrapper*   cublas_wrapper,
                           IAllocator*        allocator,
                           const bool         is_free_buffer_after_forward,
                           const bool         sparse,
                           size_t             input_channels  = 4,
                           size_t             output_channels = 4,
                           LyraQuantType      quant_level     = LyraQuantType::NONE,
                           const std::string& sd_ver          = "sd15");

                           Unet2dConditionalModel(Unet2dConditionalModel<T> const& unet);

                           virtual ~Unet2dConditionalModel();

                           virtual void forward(TensorMap*                                    output_tensors,
                                                const TensorMap*                              input_tensors,
                                                const float                                   timestep,
                                                const Unet2dConditionalModelWeight<T>*        unet_weights,
                                                const std::vector<Tensor>*                    control_imgs,
                                                const std::vector<std::vector<float>>*        controlnet_scales,
                                                const std::vector<ControlNetModelWeight<T>*>* controlnet_weights,
                                                const bool                                    controlnet_guess_mode);

                           virtual void controlnet_forward(std::vector<Tensor>&            output_tensors,
                                                           const TensorMap*                input_tensors,
                                                           const float                     timestep,
                                                           const ControlNetModelWeight<T>* weights,
                                                           const std::vector<float>&       controlnet_scales);

                           virtual void unet_forward(TensorMap*                             output_tensors,
                                                     const TensorMap*                       input_tensors,
                                                     const float                            timestep,
                                                     const Unet2dConditionalModelWeight<T>* unet_weights,
                                                     const std::vector<Tensor>&             controlnet_results);

                           virtual std::vector<std::vector<int64_t>>
                           get_controlnet_results_shape(int64_t batch_size, int64_t height, int64_t width);

                           void freeBuffer() override;
};
}  // namespace lyradiff
