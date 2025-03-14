#include "FluxSingleAttentionProcessorWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxSingleAttentionProcessorWeight<T>::FluxSingleAttentionProcessorWeight(size_t        embedding_dim,
                                                                          size_t        embedding_head_num,
                                                                          size_t        embedding_head_dim,
                                                                          LyraQuantType quant_level,
                                                                          IAllocator*   allocator)
{
    embedding_dim_      = embedding_dim;
    embedding_head_num_ = embedding_head_num;
    embedding_head_dim_ = embedding_head_dim;
    this->allocator_    = allocator;
    this->quant_level_  = quant_level;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxSingleAttentionProcessorWeight<T>::~FluxSingleAttentionProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(to_qkv_weight);
        deviceFree(to_qkv_bias);
        deviceFree(qk_norm_weight);
        if (this->quant_level_ != LyraQuantType::NONE) {
            free(to_qkv_weight_h);
        }
    }
}

template<typename T>
void FluxSingleAttentionProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&qk_norm_weight, 2 * embedding_head_dim_);

        if (this->quant_level_ != LyraQuantType::NONE) {
            to_qkv_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);

            this->lora_weight_map = {
                {"to_q", new LoraWeightV2<T>({embedding_dim_ * embedding_dim_}, to_qkv_weight, to_qkv_weight_h)},
                {"to_k",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_},
                                     &to_qkv_weight[embedding_dim_ * embedding_dim_],
                                     &to_qkv_weight_h[embedding_dim_ * embedding_dim_])},
                {"to_v",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_},
                                     &to_qkv_weight[embedding_dim_ * embedding_dim_ * 2],
                                     &to_qkv_weight_h[embedding_dim_ * embedding_dim_ * 2])},
                {"to_qkv", new LoraWeightV2<T>({embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight, to_qkv_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"to_q", new LoraWeight<T>({embedding_dim_ * embedding_dim_}, to_qkv_weight)},
                {"to_k",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_}, &to_qkv_weight[embedding_dim_ * embedding_dim_])},
                {"to_v",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_},
                                   &to_qkv_weight[embedding_dim_ * embedding_dim_ * 2])},
                {"to_qkv", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight)}};
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxSingleAttentionProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    size_t offset = 0;

    loadWeightFromBin<T>(
        &to_qkv_weight[offset], {embedding_dim_ * embedding_dim_}, prefix + "to_q.weight.bin", model_file_type);
    offset += embedding_dim_ * embedding_dim_;
    loadWeightFromBin<T>(
        &to_qkv_weight[offset], {embedding_dim_ * embedding_dim_}, prefix + "to_k.weight.bin", model_file_type);
    offset += embedding_dim_ * embedding_dim_;
    loadWeightFromBin<T>(
        &to_qkv_weight[offset], {embedding_dim_ * embedding_dim_}, prefix + "to_v.weight.bin", model_file_type);

    offset = 0;
    loadWeightFromBin<T>(&to_qkv_bias[offset], {embedding_dim_}, prefix + "to_q.bias.bin", model_file_type);
    offset += embedding_dim_;
    loadWeightFromBin<T>(&to_qkv_bias[offset], {embedding_dim_}, prefix + "to_k.bias.bin", model_file_type);
    offset += embedding_dim_;
    loadWeightFromBin<T>(&to_qkv_bias[offset], {embedding_dim_}, prefix + "to_v.bias.bin", model_file_type);

    offset = 0;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_k.weight.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(to_qkv_weight_h,
                        to_qkv_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_ * 3,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template<typename T>
void FluxSingleAttentionProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                                 std::unordered_map<std::string, void*>& weights,
                                                                 cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_q_weight = weights[prefix + "to_q.weight"];
    void* tmp_q_bias   = weights[prefix + "to_q.bias"];
    void* tmp_k_weight = weights[prefix + "to_k.weight"];
    void* tmp_k_bias   = weights[prefix + "to_k.bias"];
    void* tmp_v_weight = weights[prefix + "to_v.weight"];
    void* tmp_v_bias   = weights[prefix + "to_v.bias"];

    void* tmp_norm_q_weight = weights[prefix + "norm_q.weight"];
    void* tmp_norm_k_weight = weights[prefix + "norm_k.weight"];

    size_t offset = 0;
    cudaMemcpy(&to_qkv_weight[offset], tmp_q_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    offset += embedding_dim_ * embedding_dim_;
    cudaMemcpy(&to_qkv_weight[offset], tmp_k_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    offset += embedding_dim_ * embedding_dim_;
    cudaMemcpy(&to_qkv_weight[offset], tmp_v_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&to_qkv_bias[offset], tmp_q_bias, sizeof(T) * embedding_dim_, memcpyKind);
    offset += embedding_dim_;
    cudaMemcpy(&to_qkv_bias[offset], tmp_k_bias, sizeof(T) * embedding_dim_, memcpyKind);
    offset += embedding_dim_;
    cudaMemcpy(&to_qkv_bias[offset], tmp_v_bias, sizeof(T) * embedding_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(to_qkv_weight_h,
                        to_qkv_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_ * 3,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template class FluxSingleAttentionProcessorWeight<float>;
template class FluxSingleAttentionProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxSingleAttentionProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff