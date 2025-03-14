#include "FluxAttentionInt4ProcessorWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttentionInt4ProcessorWeight<T>::FluxAttentionInt4ProcessorWeight(size_t        embedding_dim,
                                                                      size_t        embedding_head_num,
                                                                      size_t        embedding_head_dim,
                                                                      LyraQuantType quant_level,
                                                                      IAllocator*   allocator)
{
    embedding_dim_      = embedding_dim;
    embedding_head_num_ = embedding_head_num;
    embedding_head_dim_ = embedding_head_dim;
    this->quant_level_  = quant_level;
    this->allocator_    = allocator;

    to_qkv_weight         = new W4A4GemmWeight<T>(embedding_dim * 3, embedding_dim, 32, true);
    encoder_to_qkv_weight = new W4A4GemmWeight<T>(embedding_dim * 3, embedding_dim, 32, true);
    to_out_weight         = new W4A4GemmWeight<T>(embedding_dim, embedding_dim, 32, true);
    encoder_to_out_weight = new W4A4GemmWeight<T>(embedding_dim, embedding_dim, 32, true);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxAttentionInt4ProcessorWeight<T>::~FluxAttentionInt4ProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(qk_norm_weight);
        deviceFree(encoder_qk_norm_weight);
    }
}

template<typename T>
void FluxAttentionInt4ProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&qk_norm_weight, 2 * embedding_head_dim_);
        deviceMalloc(&encoder_qk_norm_weight, 2 * embedding_head_dim_);
        to_qkv_weight->mallocWeights();
        encoder_to_qkv_weight->mallocWeights();
        to_out_weight->mallocWeights();
        encoder_to_out_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttentionInt4ProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    to_qkv_weight->loadWeights(prefix + "to_qkv.", model_file_type);
    encoder_to_qkv_weight->loadWeights(prefix + "to_added_qkv.", model_file_type);
    to_out_weight->loadWeights(prefix + "to_out.0.", model_file_type);
    encoder_to_out_weight->loadWeights(prefix + "to_add_out.", model_file_type);

    // qk_norm_weight
    int offset = 0;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_k.weight.bin", model_file_type);

    // encoder_qk_norm_weight
    offset = 0;
    loadWeightFromBin<T>(
        &encoder_qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_added_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(
        &encoder_qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_added_k.weight.bin", model_file_type);
}

template<typename T>
void FluxAttentionInt4ProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                               std::unordered_map<std::string, void*>& weights,
                                                               cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // int offset = 0;
    // cudaMemcpy(&qk_norm_weight[offset], tmp_norm_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    // offset += embedding_head_dim_;
    // cudaMemcpy(&qk_norm_weight[offset], tmp_norm_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);

    // offset = 0;
    // cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_q_weight, sizeof(T) * embedding_head_dim_,
    // memcpyKind); offset += embedding_head_dim_; cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_k_weight,
    // sizeof(T) * embedding_head_dim_, memcpyKind);
}

template class FluxAttentionInt4ProcessorWeight<float>;
template class FluxAttentionInt4ProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttentionInt4ProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff