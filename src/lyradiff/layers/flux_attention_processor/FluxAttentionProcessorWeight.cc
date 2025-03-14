#include "FluxAttentionProcessorWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttentionProcessorWeight<T>::FluxAttentionProcessorWeight(size_t        embedding_dim,
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
FluxAttentionProcessorWeight<T>::~FluxAttentionProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(to_qkv_weight);
        deviceFree(to_qkv_bias);
        deviceFree(encoder_to_qkv_weight);
        deviceFree(encoder_to_qkv_bias);
        deviceFree(qk_norm_weight);
        deviceFree(encoder_qk_norm_weight);
        deviceFree(to_out_weight);
        deviceFree(to_out_bias);
        deviceFree(encoder_to_out_weight);
        deviceFree(encoder_to_out_bias);

        if (this->quant_level_ > 0) {
            free(to_qkv_weight_h);
            free(encoder_to_qkv_weight_h);
            free(to_out_weight_h);
            free(encoder_to_out_weight_h);
        }
    }
}

template<typename T>
void FluxAttentionProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&encoder_to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&encoder_to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&qk_norm_weight, 2 * embedding_head_dim_);
        deviceMalloc(&encoder_qk_norm_weight, 2 * embedding_head_dim_);
        deviceMalloc(&to_out_weight, embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_out_bias, embedding_dim_);
        deviceMalloc(&encoder_to_out_weight, embedding_dim_ * embedding_dim_);
        deviceMalloc(&encoder_to_out_bias, embedding_dim_);

        if (this->quant_level_ > 0) {
            to_qkv_weight_h         = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);
            encoder_to_qkv_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);
            to_out_weight_h         = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_);
            encoder_to_out_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_);

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
                {"to_qkv", new LoraWeightV2<T>({embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight, to_qkv_weight_h)},
                {"to_out.0", new LoraWeightV2<T>({embedding_dim_ * embedding_dim_}, to_out_weight, to_out_weight_h)},
                {"add_q_proj",
                 new LoraWeightV2<T>(
                     {embedding_dim_ * embedding_dim_}, encoder_to_qkv_weight, encoder_to_qkv_weight_h)},
                {"add_k_proj",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_},
                                     &encoder_to_qkv_weight[embedding_dim_ * embedding_dim_],
                                     &encoder_to_qkv_weight_h[embedding_dim_ * embedding_dim_])},
                {"add_v_proj",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_},
                                     &encoder_to_qkv_weight[embedding_dim_ * embedding_dim_ * 2],
                                     &encoder_to_qkv_weight_h[embedding_dim_ * embedding_dim_ * 2])},
                {"to_added_qkv",
                 new LoraWeightV2<T>(
                     {embedding_dim_ * embedding_dim_ * 3}, encoder_to_qkv_weight, encoder_to_qkv_weight_h)},
                {"to_add_out",
                 new LoraWeightV2<T>(
                     {embedding_dim_ * embedding_dim_}, encoder_to_out_weight, encoder_to_out_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"to_q", new LoraWeight<T>({embedding_dim_ * embedding_dim_}, to_qkv_weight)},
                {"to_k",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_}, &to_qkv_weight[embedding_dim_ * embedding_dim_])},
                {"to_v",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_},
                                   &to_qkv_weight[embedding_dim_ * embedding_dim_ * 2])},
                {"to_qkv", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight)},
                {"to_out.0", new LoraWeight<T>({embedding_dim_ * embedding_dim_}, to_out_weight)},
                {"add_q_proj", new LoraWeight<T>({embedding_dim_ * embedding_dim_}, encoder_to_qkv_weight)},
                {"add_k_proj",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_},
                                   &encoder_to_qkv_weight[embedding_dim_ * embedding_dim_])},
                {"add_v_proj",
                 new LoraWeight<T>({embedding_dim_ * embedding_dim_},
                                   &encoder_to_qkv_weight[embedding_dim_ * embedding_dim_ * 2])},
                {"to_added_qkv", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * 3}, encoder_to_qkv_weight)},
                {"to_add_out", new LoraWeight<T>({embedding_dim_ * embedding_dim_}, encoder_to_out_weight)}};
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttentionProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
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
    loadWeightFromBin<T>(&encoder_to_qkv_weight[offset],
                         {embedding_dim_ * embedding_dim_},
                         prefix + "add_q_proj.weight.bin",
                         model_file_type);
    offset += embedding_dim_ * embedding_dim_;
    loadWeightFromBin<T>(&encoder_to_qkv_weight[offset],
                         {embedding_dim_ * embedding_dim_},
                         prefix + "add_k_proj.weight.bin",
                         model_file_type);
    offset += embedding_dim_ * embedding_dim_;
    loadWeightFromBin<T>(&encoder_to_qkv_weight[offset],
                         {embedding_dim_ * embedding_dim_},
                         prefix + "add_v_proj.weight.bin",
                         model_file_type);

    offset = 0;
    loadWeightFromBin<T>(
        &encoder_to_qkv_bias[offset], {embedding_dim_}, prefix + "add_q_proj.bias.bin", model_file_type);
    offset += embedding_dim_;
    loadWeightFromBin<T>(
        &encoder_to_qkv_bias[offset], {embedding_dim_}, prefix + "add_k_proj.bias.bin", model_file_type);
    offset += embedding_dim_;
    loadWeightFromBin<T>(
        &encoder_to_qkv_bias[offset], {embedding_dim_}, prefix + "add_v_proj.bias.bin", model_file_type);

    offset = 0;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(&qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_k.weight.bin", model_file_type);

    offset = 0;
    loadWeightFromBin<T>(
        &encoder_qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_added_q.weight.bin", model_file_type);
    offset += embedding_head_dim_;
    loadWeightFromBin<T>(
        &encoder_qk_norm_weight[offset], {embedding_head_dim_}, prefix + "norm_added_k.weight.bin", model_file_type);

    loadWeightFromBin<T>(
        to_out_weight, {embedding_dim_ * embedding_dim_}, prefix + "to_out.0.weight.bin", model_file_type);
    loadWeightFromBin<T>(to_out_bias, {embedding_dim_}, prefix + "to_out.0.bias.bin", model_file_type);

    loadWeightFromBin<T>(
        encoder_to_out_weight, {embedding_dim_ * embedding_dim_}, prefix + "to_add_out.weight.bin", model_file_type);
    loadWeightFromBin<T>(encoder_to_out_bias, {embedding_dim_}, prefix + "to_add_out.bias.bin", model_file_type);

    if (this->quant_level_ > 0) {
        cudaMemcpy(
            to_qkv_weight_h, to_qkv_weight, sizeof(T) * embedding_dim_ * embedding_dim_ * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(encoder_to_qkv_weight_h,
                   encoder_to_qkv_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * 3,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(to_out_weight_h, to_out_weight, sizeof(T) * embedding_dim_ * embedding_dim_, cudaMemcpyDeviceToHost);
        cudaMemcpy(encoder_to_out_weight_h,
                   encoder_to_out_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_,
                   cudaMemcpyDeviceToHost);
    }
}

template<typename T>
void FluxAttentionProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
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

    void* tmp_encoder_q_weight = weights[prefix + "add_q_proj.weight"];
    void* tmp_encoder_q_bias   = weights[prefix + "add_q_proj.bias"];
    void* tmp_encoder_k_weight = weights[prefix + "add_k_proj.weight"];
    void* tmp_encoder_k_bias   = weights[prefix + "add_k_proj.bias"];
    void* tmp_encoder_v_weight = weights[prefix + "add_v_proj.weight"];
    void* tmp_encoder_v_bias   = weights[prefix + "add_v_proj.bias"];

    void* tmp_norm_q_weight = weights[prefix + "norm_q.weight"];
    void* tmp_norm_k_weight = weights[prefix + "norm_k.weight"];

    void* tmp_norm_added_q_weight = weights[prefix + "norm_added_q.weight"];
    void* tmp_norm_added_k_weight = weights[prefix + "norm_added_k.weight"];

    void* tmp_to_out_weight = weights[prefix + "to_out.0.weight"];
    void* tmp_to_out_bias   = weights[prefix + "to_out.0.bias"];

    void* tmp_encoder_to_out_weight = weights[prefix + "to_add_out.weight"];
    void* tmp_encoder_to_out_bias   = weights[prefix + "to_add_out.bias"];

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
    cudaMemcpy(
        &encoder_to_qkv_weight[offset], tmp_encoder_q_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    offset += embedding_dim_ * embedding_dim_;
    cudaMemcpy(
        &encoder_to_qkv_weight[offset], tmp_encoder_k_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    offset += embedding_dim_ * embedding_dim_;
    cudaMemcpy(
        &encoder_to_qkv_weight[offset], tmp_encoder_v_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&encoder_to_qkv_bias[offset], tmp_encoder_q_bias, sizeof(T) * embedding_dim_, memcpyKind);
    offset += embedding_dim_;
    cudaMemcpy(&encoder_to_qkv_bias[offset], tmp_encoder_k_bias, sizeof(T) * embedding_dim_, memcpyKind);
    offset += embedding_dim_;
    cudaMemcpy(&encoder_to_qkv_bias[offset], tmp_encoder_v_bias, sizeof(T) * embedding_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);

    cudaMemcpy(to_out_weight, tmp_to_out_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    cudaMemcpy(to_out_bias, tmp_to_out_bias, sizeof(T) * embedding_dim_, memcpyKind);
    cudaMemcpy(
        encoder_to_out_weight, tmp_encoder_to_out_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind);
    cudaMemcpy(encoder_to_out_bias, tmp_encoder_to_out_bias, sizeof(T) * embedding_dim_, memcpyKind);

    if (this->quant_level_ > 0) {
        cudaMemcpyAsync(to_qkv_weight_h,
                        to_qkv_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_ * 3,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
        cudaMemcpyAsync(encoder_to_qkv_weight_h,
                        encoder_to_qkv_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_ * 3,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
        cudaMemcpyAsync(to_out_weight_h,
                        to_out_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
        cudaMemcpyAsync(encoder_to_out_weight_h,
                        encoder_to_out_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template class FluxAttentionProcessorWeight<float>;
template class FluxAttentionProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttentionProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff