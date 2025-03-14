#include "FluxTransformerFP8BlockWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxTransformerFP8BlockWeight<T>::FluxTransformerFP8BlockWeight(size_t        embedding_dim,
                                                                size_t        embedding_head_num,
                                                                size_t        embedding_head_dim,
                                                                size_t        mlp_scale,
                                                                LyraQuantType quant_level,
                                                                IAllocator*   allocator)
{
    embedding_dim_      = embedding_dim;
    embedding_head_num_ = embedding_head_num;
    embedding_head_dim_ = embedding_head_dim;
    mlp_scale_          = mlp_scale;

    this->allocator_   = allocator;
    // this->quant_level_ = quant_level;
    // quant_level = LyraQuantType::FP8_W8A8;
    this->quant_level_ = quant_level;
    // this->quant_level_ = LyraQuantType::FP8_W8A8;

    if (quant_level == LyraQuantType::FP8_W8A8_FULL) {
        ada_norm_weight         = new AdaFP8LayerNormWeight<T>(embedding_dim_, 6, quant_level, allocator);
        context_ada_norm_weight = new AdaFP8LayerNormWeight<T>(embedding_dim_, 6, quant_level, allocator);
    }
    else {
        ada_norm_weight         = new AdaLayerNormWeight<T>(embedding_dim_, 6, quant_level, allocator);
        context_ada_norm_weight = new AdaLayerNormWeight<T>(embedding_dim_, 6, quant_level, allocator);
    }

    attn_weight = new FluxAttentionFP8ProcessorWeight<T>(
        embedding_dim_, embedding_head_num_, embedding_head_dim_, quant_level, allocator);
    post_attn_weight         = new FluxAttnPostFP8ProcessorWeight<T>(embedding_dim_, quant_level, allocator);
    context_post_attn_weight = new FluxAttnPostFP8ProcessorWeight<T>(embedding_dim_, quant_level, allocator);

    this->lora_layer_map = {{"norm1", ada_norm_weight},
                            {"norm1_context", context_ada_norm_weight},
                            {"attn", attn_weight},
                            {"ff", post_attn_weight},
                            {"ff_context", context_post_attn_weight}};

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxTransformerFP8BlockWeight<T>::~FluxTransformerFP8BlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        delete ada_norm_weight;
        delete context_ada_norm_weight;
        delete attn_weight;
        delete post_attn_weight;
        delete context_post_attn_weight;
    }
}

template<typename T>
void FluxTransformerFP8BlockWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        ada_norm_weight->mallocWeights();
        context_ada_norm_weight->mallocWeights();
        attn_weight->mallocWeights();
        post_attn_weight->mallocWeights();
        context_post_attn_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxTransformerFP8BlockWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeights(prefix + "norm1.", model_file_type);
    context_ada_norm_weight->loadWeights(prefix + "norm1_context.", model_file_type);
    attn_weight->loadWeights(prefix + "attn.", model_file_type);
    post_attn_weight->loadWeights(prefix + "ff.", model_file_type);
    context_post_attn_weight->loadWeights(prefix + "ff_context.", model_file_type);
}

template<typename T>
void FluxTransformerFP8BlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                            std::unordered_map<std::string, void*>& weights,
                                                            cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeightsFromCache(prefix + "norm1.", weights, memcpyKind);
    context_ada_norm_weight->loadWeightsFromCache(prefix + "norm1_context.", weights, memcpyKind);
    attn_weight->loadWeightsFromCache(prefix + "attn.", weights, memcpyKind);
    post_attn_weight->loadWeightsFromCache(prefix + "ff.", weights, memcpyKind);
    context_post_attn_weight->loadWeightsFromCache(prefix + "ff_context.", weights, memcpyKind);
}

template class FluxTransformerFP8BlockWeight<float>;
template class FluxTransformerFP8BlockWeight<half>;

#ifdef ENABLE_BF16
template class FluxTransformerFP8BlockWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff