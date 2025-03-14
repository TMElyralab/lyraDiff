#include "ZeroCrossAttnWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ZeroCrossAttnWeight<T>::ZeroCrossAttnWeight(size_t query_dim, size_t context_dim)
{
    query_dim_   = query_dim;
    context_dim_ = context_dim;
    heads_       = query_dim_ / dim_head_;
    inner_dim_   = heads_ * dim_head_;

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ZeroCrossAttnWeight<T>::~ZeroCrossAttnWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(attention_q_weight);
        deviceFree(attention_kv_weight);
        deviceFree(attention_to_out_weight);
        deviceFree(attention_to_out_bias);
        deviceFree(norm1_gamma);
        deviceFree(norm1_beta);
        deviceFree(norm2_gamma);
        deviceFree(norm2_beta);
    }
}

template<typename T>
void ZeroCrossAttnWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&attention_q_weight, query_dim_ * inner_dim_);

        deviceMalloc(&attention_kv_weight, context_dim_ * 2 * inner_dim_);

        deviceMalloc(&attention_to_out_weight, inner_dim_ * query_dim_);
        deviceMalloc(&attention_to_out_bias, query_dim_);

        deviceMalloc(&norm1_gamma, query_dim_);
        deviceMalloc(&norm1_beta, query_dim_);

        deviceMalloc(&norm2_gamma, context_dim_);
        deviceMalloc(&norm2_beta, context_dim_);

        is_maintain_buffer = true;
    }
}

template<typename T>
void ZeroCrossAttnWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    loadWeightFromBin<T>(
        attention_q_weight, {query_dim_ * inner_dim_}, prefix + "attn.to_q.weight.bin", model_file_type);

    int offset = 0;
    loadWeightFromBin<T>(
        &attention_kv_weight[offset], {context_dim_, inner_dim_}, prefix + "attn.to_k.weight.bin", model_file_type);
    offset += context_dim_ * inner_dim_;
    loadWeightFromBin<T>(
        &attention_kv_weight[offset], {context_dim_, inner_dim_}, prefix + "attn.to_v.weight.bin", model_file_type);

    loadWeightFromBin<T>(
        attention_to_out_weight, {inner_dim_ * query_dim_}, prefix + "attn.to_out.0.weight.bin", model_file_type);
    loadWeightFromBin<T>(attention_to_out_bias, {query_dim_}, prefix + "attn.to_out.0.bias.bin", model_file_type);

    loadWeightFromBin<T>(norm1_gamma, {query_dim_}, prefix + "norm1.weight.bin", model_file_type);
    loadWeightFromBin<T>(norm1_beta, {query_dim_}, prefix + "norm1.bias.bin", model_file_type);

    loadWeightFromBin<T>(norm2_gamma, {context_dim_}, prefix + "norm2.weight.bin", model_file_type);
    loadWeightFromBin<T>(norm2_beta, {context_dim_}, prefix + "norm2.bias.bin", model_file_type);
}

template<typename T>
void ZeroCrossAttnWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                  std::unordered_map<std::string, void*>& weights,
                                                  cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_attn_to_q_weights = weights[prefix + "attn.to_q.weight"];
    void* tmp_attn_to_k_weights = weights[prefix + "attn.to_k.weight"];
    void* tmp_attn_to_v_weights = weights[prefix + "attn.to_v.weight"];

    void* tmp_attn_to_out_weights = weights[prefix + "attn.to_out.0.weight"];
    void* tmp_attn_to_out_bias    = weights[prefix + "attn.to_out.0.bias"];
    void* tmp_norm1_weights       = weights[prefix + "norm1.weight"];
    void* tmp_norm1_bias          = weights[prefix + "norm1.bias"];
    void* tmp_norm2_weights       = weights[prefix + "norm2.weight"];
    void* tmp_norm2_bias          = weights[prefix + "norm2.bias"];

    cudaMemcpy(attention_q_weight, tmp_attn_to_q_weights, sizeof(T) * query_dim_ * inner_dim_, memcpyKind);

    int offset = 0;
    cudaMemcpy(&attention_kv_weight[offset], tmp_attn_to_k_weights, sizeof(T) * context_dim_ * inner_dim_, memcpyKind);
    offset += context_dim_ * inner_dim_;
    cudaMemcpy(&attention_kv_weight[offset], tmp_attn_to_v_weights, sizeof(T) * context_dim_ * inner_dim_, memcpyKind);

    cudaMemcpy(attention_to_out_weight, tmp_attn_to_out_weights, sizeof(T) * inner_dim_ * query_dim_, memcpyKind);
    cudaMemcpy(attention_to_out_bias, tmp_attn_to_out_bias, sizeof(T) * query_dim_, memcpyKind);

    cudaMemcpy(norm1_gamma, tmp_norm1_weights, sizeof(T) * query_dim_, memcpyKind);
    cudaMemcpy(norm1_beta, tmp_norm1_bias, sizeof(T) * query_dim_, memcpyKind);

    cudaMemcpy(norm2_gamma, tmp_norm2_weights, sizeof(T) * context_dim_, memcpyKind);
    cudaMemcpy(norm2_beta, tmp_norm2_bias, sizeof(T) * context_dim_, memcpyKind);
}

template class ZeroCrossAttnWeight<float>;
template class ZeroCrossAttnWeight<half>;
}  // namespace lyradiff