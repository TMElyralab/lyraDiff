#include "CombinedTimestepGuidanceTextProjEmbeddingsWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>::CombinedTimestepGuidanceTextProjEmbeddingsWeight(
    const size_t        pooled_projection_dim,
    const size_t        embedding_dim,
    const size_t        embedding_input_dim,
    const LyraQuantType quant_level,
    IAllocator*         allocator):
    pooled_projection_dim_(pooled_projection_dim),
    embedding_dim_(embedding_dim),
    embedding_input_dim_(embedding_input_dim)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    timestep_emb_weight = new TimestepEmbeddingBlockWeight<T>(
        embedding_input_dim_, embedding_dim_, embedding_dim_, quant_level, allocator);
    guidance_emb_weight = new TimestepEmbeddingBlockWeight<T>(
        embedding_input_dim_, embedding_dim_, embedding_dim_, quant_level, allocator);
    text_emb_weight = new TimestepEmbeddingBlockWeight<T>(
        pooled_projection_dim_, embedding_dim_, embedding_dim_, quant_level, allocator);

    this->allocator_     = allocator;
    this->quant_level_   = quant_level;
    this->lora_layer_map = {{"guidance_embedder", guidance_emb_weight},
                            {"text_embedder", text_emb_weight},
                            {"timestep_embedder", timestep_emb_weight}};
}

template<typename T>
CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>::~CombinedTimestepGuidanceTextProjEmbeddingsWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    timestep_emb_weight->mallocWeights();
    guidance_emb_weight->mallocWeights();
    text_emb_weight->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>::loadWeights(std::string    prefix,
                                                                      FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    timestep_emb_weight->loadWeights(prefix + "timestep_embedder.", model_file_type);
    guidance_emb_weight->loadWeights(prefix + "guidance_embedder.", model_file_type);
    text_emb_weight->loadWeights(prefix + "text_embedder.", model_file_type);
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>::loadWeightsFromCache(
    std::string prefix, std::unordered_map<std::string, void*>& weights, cudaMemcpyKind memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    timestep_emb_weight->loadWeightsFromCache(prefix + "timestep_embedder.", weights, memcpyKind);
    guidance_emb_weight->loadWeightsFromCache(prefix + "guidance_embedder.", weights, memcpyKind);
    text_emb_weight->loadWeightsFromCache(prefix + "text_embedder.", weights, memcpyKind);
}

template class CombinedTimestepGuidanceTextProjEmbeddingsWeight<float>;
template class CombinedTimestepGuidanceTextProjEmbeddingsWeight<half>;
#ifdef ENABLE_BF16
template class CombinedTimestepGuidanceTextProjEmbeddingsWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff
