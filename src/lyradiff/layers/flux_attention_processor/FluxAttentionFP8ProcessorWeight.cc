#include "FluxAttentionFP8ProcessorWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttentionFP8ProcessorWeight<T>::FluxAttentionFP8ProcessorWeight(size_t        embedding_dim,
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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxAttentionFP8ProcessorWeight<T>::~FluxAttentionFP8ProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(to_qkv_weight);
        deviceFree(to_qkv_bias);
        deviceFree(to_qkv_weight_scale);
        deviceFree(to_qkv_input_scale);

        deviceFree(encoder_to_qkv_weight);
        deviceFree(encoder_to_qkv_bias);
        deviceFree(encoder_to_qkv_weight_scale);
        deviceFree(encoder_to_qkv_input_scale);

        deviceFree(qk_norm_weight);
        deviceFree(encoder_qk_norm_weight);

        deviceFree(to_out_weight);
        deviceFree(to_out_bias);
        deviceFree(to_out_weight_scale);
        deviceFree(to_out_input_scale);

        deviceFree(encoder_to_out_weight);
        deviceFree(encoder_to_out_bias);
        deviceFree(encoder_to_out_weight_scale);
        deviceFree(encoder_to_out_input_scale);

        free(to_qkv_weight_h);
        free(encoder_to_qkv_weight_h);
        free(to_out_weight_h);
        free(encoder_to_out_weight_h);
    }
}

template<typename T>
void FluxAttentionFP8ProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&to_qkv_weight_scale, 1);
        deviceMalloc(&to_qkv_input_scale, 1);

        deviceMalloc(&encoder_to_qkv_weight, 3 * embedding_dim_ * embedding_dim_);
        deviceMalloc(&encoder_to_qkv_bias, 3 * embedding_dim_);
        deviceMalloc(&encoder_to_qkv_weight_scale, 1);
        deviceMalloc(&encoder_to_qkv_input_scale, 1);

        deviceMalloc(&qk_norm_weight, 2 * embedding_head_dim_);
        deviceMalloc(&encoder_qk_norm_weight, 2 * embedding_head_dim_);

        deviceMalloc(&to_out_weight, embedding_dim_ * embedding_dim_);
        deviceMalloc(&to_out_bias, embedding_dim_);
        deviceMalloc(&to_out_weight_scale, 1);
        deviceMalloc(&to_out_input_scale, 1);

        deviceMalloc(&encoder_to_out_weight, embedding_dim_ * embedding_dim_);
        deviceMalloc(&encoder_to_out_bias, embedding_dim_);
        deviceMalloc(&encoder_to_out_weight_scale, 1);
        deviceMalloc(&encoder_to_out_input_scale, 1);

        to_qkv_weight_h         = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);
        encoder_to_qkv_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 3);
        to_out_weight_h         = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_);
        encoder_to_out_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_);

        this->lora_weight_map = {
            {"to_qkv",
             new FP8LoraWeight<T>(
                 {embedding_dim_ * embedding_dim_ * 3}, to_qkv_weight, to_qkv_weight_scale, to_qkv_weight_h)},
            {"to_out.0",
             new FP8LoraWeight<T>(
                 {embedding_dim_ * embedding_dim_}, to_out_weight, to_out_weight_scale, to_out_weight_h)},
            {"to_added_qkv",
             new FP8LoraWeight<T>({embedding_dim_ * embedding_dim_ * 3},
                                  encoder_to_qkv_weight,
                                  encoder_to_qkv_weight_scale,
                                  encoder_to_qkv_weight_h)},
            {"to_add_out",
             new FP8LoraWeight<T>({embedding_dim_ * embedding_dim_},
                                  encoder_to_out_weight,
                                  encoder_to_out_weight_scale,
                                  encoder_to_out_weight_h)}};
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttentionFP8ProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    size_t offset       = 0;
    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 3;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromBin(to_qkv, cur_size, cur_size_2, "to_qkv");
    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromBin(encoder_to_qkv, cur_size, cur_size_2, "to_added_qkv");
    cur_size   = embedding_dim_ * embedding_dim_;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromBin(to_out, cur_size, cur_size_2, "to_out.0");
    cur_size   = embedding_dim_ * embedding_dim_;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromBin(encoder_to_out, cur_size, cur_size_2, "to_add_out");

    // to_qkv_weight
    // loadWeightFromBin<__nv_fp8_e4m3>(
    //     to_qkv_weight, {embedding_dim_ * embedding_dim_ * 3}, prefix + "to_qkv_weight_fp8.bin", model_file_type);
    // loadWeightFromBin<T>(to_qkv_bias, {embedding_dim_ * 3}, prefix + "to_qkv.bias.bin", model_file_type);

    // loadWeightFromBin<float>(to_qkv_weight_scale, {1}, prefix + "to_qkv_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(to_qkv_input_scale, {1}, prefix + "to_qkv_input_scale.bin", FtCudaDataType::FP32);

    // encoder_to_qkv_weight
    // loadWeightFromBin<__nv_fp8_e4m3>(encoder_to_qkv_weight,
    //                                  {embedding_dim_ * embedding_dim_ * 3},
    //                                  prefix + "to_added_qkv_weight_fp8.bin",
    //                                  model_file_type);
    // loadWeightFromBin<T>(encoder_to_qkv_bias, {embedding_dim_ * 3}, prefix + "to_added_qkv.bias.bin",
    // model_file_type); loadWeightFromBin<float>(
    //     encoder_to_qkv_weight_scale, {1}, prefix + "to_added_qkv_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(
    //     encoder_to_qkv_input_scale, {1}, prefix + "to_added_qkv_input_scale.bin", FtCudaDataType::FP32);

    // qk_norm_weight
    offset = 0;
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

    // encoder_to_out_weight
    // loadWeightFromBin<__nv_fp8_e4m3>(
    //     to_out_weight, {embedding_dim_ * embedding_dim_}, prefix + "to_out.0_weight_fp8.bin", model_file_type);
    // loadWeightFromBin<T>(to_out_bias, {embedding_dim_}, prefix + "to_out.0.bias.bin", model_file_type);

    // loadWeightFromBin<float>(to_out_weight_scale, {1}, prefix + "to_out.0_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(to_out_input_scale, {1}, prefix + "to_out.0_input_scale.bin", FtCudaDataType::FP32);

    // encoder_to_out_weight
    // loadWeightFromBin<__nv_fp8_e4m3>(encoder_to_out_weight,
    //                                  {embedding_dim_ * embedding_dim_},
    //                                  prefix + "to_add_out_weight_fp8.bin",
    //                                  model_file_type);
    // loadWeightFromBin<T>(encoder_to_out_bias, {embedding_dim_}, prefix + "to_add_out.bias.bin", model_file_type);

    // loadWeightFromBin<float>(
    //     encoder_to_out_weight_scale, {1}, prefix + "to_add_out_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(
    //     encoder_to_out_input_scale, {1}, prefix + "to_add_out_input_scale.bin", FtCudaDataType::FP32);
}

template<typename T>
void FluxAttentionFP8ProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                              std::unordered_map<std::string, void*>& weights,
                                                              cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    size_t offset       = 0;
    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 3;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromCache(to_qkv, cur_size, cur_size_2, "to_qkv");
    cur_size   = embedding_dim_ * embedding_dim_ * 3;
    cur_size_2 = embedding_dim_ * 3;
    MACROLoadFP8WeightFromCache(encoder_to_qkv, cur_size, cur_size_2, "to_added_qkv");
    cur_size   = embedding_dim_ * embedding_dim_;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromCache(to_out, cur_size, cur_size_2, "to_out.0");
    cur_size   = embedding_dim_ * embedding_dim_;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromCache(encoder_to_out, cur_size, cur_size_2, "to_add_out");

    void* tmp_norm_q_weight = weights[prefix + "norm_q.weight"];
    void* tmp_norm_k_weight = weights[prefix + "norm_k.weight"];

    void* tmp_norm_added_q_weight = weights[prefix + "norm_added_q.weight"];
    void* tmp_norm_added_k_weight = weights[prefix + "norm_added_k.weight"];

    offset = 0;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&qk_norm_weight[offset], tmp_norm_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);

    offset = 0;
    cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_q_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
    offset += embedding_head_dim_;
    cudaMemcpy(&encoder_qk_norm_weight[offset], tmp_norm_added_k_weight, sizeof(T) * embedding_head_dim_, memcpyKind);
}

template class FluxAttentionFP8ProcessorWeight<float>;
template class FluxAttentionFP8ProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttentionFP8ProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff