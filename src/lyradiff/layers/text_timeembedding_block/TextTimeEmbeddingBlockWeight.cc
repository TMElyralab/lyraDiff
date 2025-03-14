#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlockWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
TextTimeEmbeddingBlockWeight<T>::TextTimeEmbeddingBlockWeight(const size_t        timestep_input_dim,
                                                              const size_t        augemb_input_dim,
                                                              const size_t        output_dim_0,
                                                              const size_t        output_dim,
                                                              const LyraQuantType quant_level,
                                                              IAllocator*         allocator):
    timestep_input_dim_(timestep_input_dim),
    augemb_input_dim_(augemb_input_dim),
    output_dim_0_(output_dim_0),
    output_dim_(output_dim)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_   = allocator;
    this->quant_level_ = quant_level;
    timestep_emb_weight =
        new TimestepEmbeddingBlockWeight<T>(timestep_input_dim_, output_dim_0_, output_dim_, quant_level, allocator);
    augemb_weight =
        new TimestepEmbeddingBlockWeight<T>(augemb_input_dim_, output_dim_0_, output_dim_, quant_level, allocator);
}

template<typename T>
TextTimeEmbeddingBlockWeight<T>::~TextTimeEmbeddingBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void TextTimeEmbeddingBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    timestep_emb_weight->mallocWeights();
    augemb_weight->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void TextTimeEmbeddingBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    timestep_emb_weight->loadWeights(prefix + "time_embedding.", model_file_type);
    augemb_weight->loadWeights(prefix + "add_embedding.", model_file_type);
}

template<typename T>
void TextTimeEmbeddingBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                           std::unordered_map<std::string, void*>& weights,
                                                           cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    timestep_emb_weight->loadWeightsFromCache("time_embedding.", weights, memcpyKind);
    augemb_weight->loadWeightsFromCache("add_embedding.", weights, memcpyKind);
}

template class TextTimeEmbeddingBlockWeight<float>;
template class TextTimeEmbeddingBlockWeight<half>;
}  // namespace lyradiff
