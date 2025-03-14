#pragma once

#include "FluxTransformer2DModelWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNorm.h"
#include "src/lyradiff/layers/combined_timestep_guidance_text_proj_embeddings_block/CombinedTimestepGuidanceTextProjEmbeddings.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerBlock.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerFP8Block.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerInt4Block.h"
#include "src/lyradiff/layers/flux_transformer_block/FluxTransformerBlock.h"
#include "src/lyradiff/layers/flux_transformer_block/FluxTransformerFP8Block.h"
#include "src/lyradiff/layers/flux_transformer_block/FluxTransformerInt4Block.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxTransformer2DModel: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t input_channels_        = 64;
    size_t num_layers_            = 19;
    size_t num_single_layers_     = 38;
    size_t attention_head_dim_    = 128;
    size_t num_attention_heads_   = 24;
    size_t pooled_projection_dim_ = 768;
    size_t joint_attention_dim_   = 4096;
    size_t embedding_input_dim_   = 256;  // 写死在 flux 代码里的，不需要入参
    size_t embedding_dim_         = 3072;
    bool   guidance_embeds_       = true;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t encoder_seq_len);

public:
    T* temb_buf_ = nullptr;
    // T* temb_buf_     = nullptr;
    T* hidden_buf_1  = nullptr;
    T* hidden_buf_2  = nullptr;
    T* encoder_buf_1 = nullptr;
    T* encoder_buf_2 = nullptr;

    T* encoder_buf_ = nullptr;

    T* msa_buf_ = nullptr;

    T* cat_hidden_buf_1 = nullptr;
    T* cat_hidden_buf_2 = nullptr;

    // std::vector<T*> down_hidden_states_bufs_ = std::vector<T*>(11, nullptr);
    // std::vector<T*> up_hidden_states_bufs_   = std::vector<T*>(4, nullptr);
    // std::vector<T*> controlnet_res_bufs_     = std::vector<T*>(13, nullptr);

    // T*      mid_hidden_res_buf_ = nullptr;
    // double* norm_cache_buf_     = nullptr;

    // std::vector<size_t> height_bufs_         = {0, 0, 0, 0};
    // std::vector<size_t> width_bufs_          = {0, 0, 0, 0};
    // size_t              cur_batch            = 0;
    // size_t              cur_controlnet_count = 0;

    CombinedTimestepGuidanceTextProjEmbeddings<T>* timestep_embedding       = nullptr;
    FluxSingleTransformerBlock<T>*                 single_transformer_block = nullptr;
    FluxTransformerBlock<T>*                       transformer_block        = nullptr;
    AdaLayerNorm<T>*                               norm_out                 = nullptr;

    FluxTransformer2DModel(cudaStream_t        stream,
                           cublasMMWrapper*    cublas_wrapper,
                           IAllocator*         allocator,
                           const bool          is_free_buffer_after_forward,
                           const bool          sparse,
                           const size_t        input_channels        = 64,
                           const size_t        num_layers            = 19,
                           const size_t        num_single_layers     = 38,
                           const size_t        attention_head_dim    = 128,
                           const size_t        num_attention_heads   = 24,
                           const size_t        pooled_projection_dim = 768,
                           const size_t        joint_attention_dim   = 4096,
                           const bool          guidance_embeds       = true,
                           const LyraQuantType quant_level           = LyraQuantType::NONE);

    FluxTransformer2DModel(FluxTransformer2DModel<T> const& other);

    virtual ~FluxTransformer2DModel();

    // virtual void forward(TensorMap*                                    output_tensors,
    //                      const TensorMap*                              input_tensors,
    //                      const float                                   timestep,
    //                      const Unet2dConditionalModelWeight<T>*        unet_weights,
    //                      const std::vector<Tensor>*                    control_imgs,
    //                      const std::vector<std::vector<float>>*        controlnet_scales,
    //                      const std::vector<ControlNetModelWeight<T>*>* controlnet_weights,
    //                      const bool                                    controlnet_guess_mode);

    virtual void transformer_forward(TensorMap*                             output_tensors,
                                     const TensorMap*                       input_tensors,
                                     const float                            timestep,
                                     const float                            guidance,
                                     const FluxTransformer2DModelWeight<T>* weights,
                                     const std::vector<Tensor>&             controlnet_block_samples,
                                     const std::vector<Tensor>&             controlnet_single_block_samples,
                                     const bool                             controlnet_blocks_repeat);

    void freeBuffer() override;
};
}  // namespace lyradiff
