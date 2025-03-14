#pragma once
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNormWeight.h"
#include "src/lyradiff/layers/combined_timestep_guidance_text_proj_embeddings_block/CombinedTimestepGuidanceTextProjEmbeddingsWeight.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerBlockWeight.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerFP8BlockWeight.h"
#include "src/lyradiff/layers/flux_transformer_block/FluxTransformerBlockWeight.h"
#include "src/lyradiff/layers/flux_transformer_block/FluxTransformerFP8BlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>
#include <vector>

namespace lyradiff {
template<typename T>
class FluxControlnetModelWeight: public IFLoraWeight<T> {
private:
    size_t input_channels_        = 64;
    size_t num_layers_            = 2;
    size_t num_single_layers_     = 0;
    size_t attention_head_dim_    = 128;
    size_t num_attention_heads_   = 24;
    size_t pooled_projection_dim_ = 768;
    size_t joint_attention_dim_   = 4096;
    size_t embedding_input_dim_   = 256;  // 写死在 flux 代码里的，不需要入参
    size_t embedding_dim_         = 3072;

protected:
    bool guidance_embeds_   = true;
    bool is_maintain_buffer = false;
    bool is_malloced        = false;

public:
    T* x_embedder_weight = nullptr;
    T* x_embedder_bias   = nullptr;

    T* context_embedder_weight = nullptr;
    T* context_embedder_bias   = nullptr;

    T* controlnet_x_embedder_weight = nullptr;
    T* controlnet_x_embedder_bias   = nullptr;

    // T* proj_out_weight = nullptr;
    // T* proj_out_bias   = nullptr;

    CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>* timestep_embedding_weight;
    std::vector<FluxSingleTransformerBlockWeight<T>*>    single_transformer_block_weight;
    std::vector<FluxTransformerBlockWeight<T>*>          transformer_block_weight;

    std::vector<T*> control_weights;
    std::vector<T*> control_bias;

    FluxControlnetModelWeight(const size_t        input_channels        = 64,
                              const size_t        num_layers            = 2,
                              const size_t        num_single_layers     = 0,
                              const size_t        attention_head_dim    = 128,
                              const size_t        num_attention_heads   = 24,
                              const size_t        pooled_projection_dim = 768,
                              const size_t        joint_attention_dim   = 4096,
                              const bool          guidance_embeds       = true,
                              const LyraQuantType quant_level           = LyraQuantType::NONE,
                              IAllocator*         allocator             = nullptr);

    ~FluxControlnetModelWeight();

    virtual void   updateConfig(const size_t num_layers = 2, const size_t num_single_layers = 0);
    virtual size_t getNumLayers();
    virtual size_t getNumSingleLayers();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
