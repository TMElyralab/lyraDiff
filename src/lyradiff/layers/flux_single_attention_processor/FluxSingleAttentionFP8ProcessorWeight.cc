#include "FluxSingleAttentionFP8ProcessorWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxSingleAttentionFP8ProcessorWeight<T>::FluxSingleAttentionFP8ProcessorWeight(size_t        embedding_dim,
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
FluxSingleAttentionFP8ProcessorWeight<T>::~FluxSingleAttentionFP8ProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(to_qkv_weight);
        deviceFree(to_qkv_bias);
        deviceFree(qk_norm_weight);
        deviceFree(to_qkv_weight_scale);
        deviceFree(to_qkv_input_scale);

        free(to_qkv_weight_h);
    }
}

template<typename T>
void FluxSingleAttentionFP8ProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&qk_norm_weight, 2 * embedding_head_dim_);
        deviceMalloc(&to_qkv_weight_scale, 1);
        deviceMalloc(&to_qkv_input_scale, 1);

        to_qkv_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);

        this->lora_weight_map = {
            {"to_qkv",
             new FP8LoraWeight<T>(
                 {embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight, to_qkv_weight_scale, to_qkv_weight_h)}};
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxSingleAttentionFP8ProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    size_t offset = 0;

    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 3;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromBin(to_qkv, cur_size, cur_size_2, "to_qkv");

    // loadWeightFromBin<__nv_fp8_e4m3>(
    //     to_qkv_weight, {embedding_dim_ * embedding_dim_ * 3}, prefix + "to_qkv_weight_fp8.bin", model_file_type);

    // loadWeightFromBin<T>(to_qkv_bias, {embedding_dim_ * 3}, prefix + "to_qkv.bias.bin", model_file_type);

    // loadWeightFromBin<float>(to_qkv_weight_scale, {1}, prefix + "to_qkv_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(to_qkv_input_scale, {1}, prefix + "to_qkv_input_scale.bin", FtCudaDataType::FP32);

    offset = 0;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_k.weight.bin", model_file_type);
}

template<typename T>
void FluxSingleAttentionFP8ProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                                    std::unordered_map<std::string, void*>& weights,
                                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 3;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromCache(to_qkv, cur_size, cur_size_2, "to_qkv");

    void* tmp_norm_q_weight = weights[prefix + "norm_q.weight"];
    void* tmp_norm_k_weight = weights[prefix + "norm_k.weight"];

    size_t offset = 0;

    offset = 0;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
}

template class FluxSingleAttentionFP8ProcessorWeight<float>;
template class FluxSingleAttentionFP8ProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxSingleAttentionFP8ProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff