#include "src/lyradiff/layers/transformer2d/Transformer2dBlockWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {

template<typename T>
Transformer2dBlockWeight<T>::Transformer2dBlockWeight(size_t        in_channels,
                                                      size_t        head_num,
                                                      size_t        dim_per_head,
                                                      size_t        cross_attn_dim,
                                                      size_t        norm_num_groups,
                                                      LyraQuantType quant_level,
                                                      IAllocator*   allocator):
    in_channels_(in_channels),
    head_num_(head_num),
    dim_per_head_(dim_per_head),
    inner_dim_(head_num * dim_per_head),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // sd21
    if (cross_attn_dim == 1024) {
        head_num_     = inner_dim_ / 64;
        dim_per_head_ = 64;
    }

    proj_in_size_  = inner_dim_ * 1 * 1 * in_channels_;
    proj_out_size_ = in_channels_ * 1 * 1 * inner_dim_;

    this->quant_level_ = quant_level;
    this->allocator_   = allocator;
    if (this->quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block_weight = new BasicTransformerInt8BlockWeight<T>(
            inner_dim_, head_num_, dim_per_head_, cross_attn_dim_, quant_level, allocator);
    }
    else {
        basic_transformer_block_weight =
            new BasicTransformerBlockWeight<T>(inner_dim_, head_num_, dim_per_head_, cross_attn_dim_, allocator);
    }

    this->vec_basic_transformer_container_weights.emplace_back(basic_transformer_block_weight);

    this->lora_layer_map = {{"transformer_blocks_0", basic_transformer_block_weight}};
}

template<typename T>
Transformer2dBlockWeight<T>::~Transformer2dBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(norm_gamma);
        deviceFree(norm_beta);
        deviceFree(proj_in_weight);
        deviceFree(proj_in_bias);
        deviceFree(proj_out_weight);
        deviceFree(proj_out_bias);

        norm_gamma = nullptr;
        norm_beta  = nullptr;

        proj_in_weight = nullptr;
        proj_in_bias   = nullptr;

        proj_out_weight = nullptr;
        proj_out_bias   = nullptr;

        if (this->quant_level_ != LyraQuantType::NONE) {
            delete basic_transformer_int8_block_weight;
            basic_transformer_int8_block_weight = nullptr;
        }
        else {
            delete basic_transformer_block_weight;
            basic_transformer_block_weight = nullptr;
        }
    }

    if (is_maintain_lora) {
        deviceFree(proj_in_lora_buffer);
        deviceFree(proj_out_lora_buffer);
    }
}

template<typename T>
void Transformer2dBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        deviceMalloc(&norm_gamma, in_channels_);
        deviceMalloc(&norm_beta, in_channels_);

        deviceMalloc(&proj_in_weight, proj_in_size_);
        deviceMalloc(&proj_in_bias, inner_dim_);

        deviceMalloc(&proj_out_weight, proj_out_size_);
        deviceMalloc(&proj_out_bias, in_channels_);

        // this->lora_weight_size_map = {{"proj_in", proj_in_size_}, {"proj_out", proj_out_size_}};
        // this->lora_weight_map      = {{"proj_in", proj_in_weight}, {"proj_out", proj_out_weight}};
        this->lora_weight_map = {{"proj_in", new LoraWeight<T>({inner_dim_, 1, 1, in_channels_}, proj_in_weight)},
                                 {"proj_out", new LoraWeight<T>({in_channels_, 1, 1, inner_dim_}, proj_out_weight)}};

        if (this->quant_level_ != LyraQuantType::NONE) {
            basic_transformer_int8_block_weight->mallocWeights();
        }
        else {
            basic_transformer_block_weight->mallocWeights();
        }
        // mallocLoraBuffer();
    }
    is_maintain_buffer = true;
}

template<typename T>
void Transformer2dBlockWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_lora) {
        deviceMalloc(&proj_in_lora_buffer, proj_in_size_);
        deviceMalloc(&proj_out_lora_buffer, proj_out_size_);
    }
    is_maintain_lora = true;
}

template<typename T>
void Transformer2dBlockWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                     std::string                          prefix,
                                                     std::unordered_map<std::string, T*>& lora_weights,
                                                     float                                lora_alpha,
                                                     FtCudaDataType                       lora_file_type,
                                                     cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }

    // 先从本地 load 到显存
    loadWeightFromBin<T>(proj_in_lora_buffer,
                         {in_channels_ * 1 * 1 * inner_dim_},
                         lora_path + prefix + std::string("proj_in.bin"),
                         lora_file_type);
    loadWeightFromBin<T>(proj_out_lora_buffer,
                         {in_channels_ * 1 * 1 * inner_dim_},
                         lora_path + prefix + std::string("proj_out.bin"),
                         lora_file_type);

    T* proj_in  = (T*)malloc(sizeof(T) * proj_in_size_);
    T* proj_out = (T*)malloc(sizeof(T) * proj_out_size_);

    // 再缓存到本地
    cudaMemcpy(proj_in, proj_in_lora_buffer, sizeof(T) * proj_in_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(proj_out, proj_out_lora_buffer, sizeof(T) * proj_out_size_, cudaMemcpyDeviceToHost);

    lora_weights[prefix + "proj_in"]  = proj_in;
    lora_weights[prefix + "proj_out"] = proj_out;

    invokeLoadLora<T>(proj_in_weight, proj_in_lora_buffer, proj_in_size_, lora_alpha, stream);
    invokeLoadLora<T>(proj_out_weight, proj_out_lora_buffer, proj_out_size_, lora_alpha, stream);

    if (this->quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block_weight->loadLoraFromWeight(
            lora_path, prefix + "transformer_blocks_0_", lora_weights, lora_alpha, lora_file_type);
    }
    else {
        basic_transformer_block_weight->loadLoraFromWeight(
            lora_path, prefix + "transformer_blocks_0_", lora_weights, lora_alpha, lora_file_type);
    }
}

template<typename T>
void Transformer2dBlockWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                    std::unordered_map<std::string, T*>& lora_weights,
                                                    float                                lora_alpha,
                                                    bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }
    T* proj_in  = lora_weights[prefix + "proj_in"];
    T* proj_out = lora_weights[prefix + "proj_out"];

    cudaMemcpy(proj_in_lora_buffer, proj_in, sizeof(T) * proj_in_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(proj_out_lora_buffer, proj_out, sizeof(T) * proj_out_size_, cudaMemcpyHostToDevice);

    invokeLoadLora<T>(proj_in_weight, proj_in_lora_buffer, proj_in_size_, lora_alpha);
    invokeLoadLora<T>(proj_out_weight, proj_out_lora_buffer, proj_out_size_, lora_alpha);

    basic_transformer_block_weight->loadLoraFromCache(
        prefix + "transformer_blocks_0_", lora_weights, lora_alpha, from_outside);
}

template<typename T>
void Transformer2dBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    loadWeightFromBin<T>(norm_gamma, {in_channels_}, prefix + "norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(norm_beta, {in_channels_}, prefix + "norm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        proj_in_weight, {inner_dim_ * 1 * 1 * in_channels_}, prefix + "proj_in.weight.bin", model_file_type);
    loadWeightFromBin<T>(proj_in_bias, {inner_dim_}, prefix + "proj_in.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        proj_out_weight, {in_channels_ * 1 * 1 * inner_dim_}, prefix + "proj_out.weight.bin", model_file_type);
    loadWeightFromBin<T>(proj_out_bias, {in_channels_}, prefix + "proj_out.bias.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block_weight->loadWeights(prefix + std::string("transformer_blocks.0."),
                                                         model_file_type);
    }
    else {
        basic_transformer_block_weight->loadWeights(prefix + std::string("transformer_blocks.0."), model_file_type);
    }
}

template<typename T>
void Transformer2dBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                       std::unordered_map<std::string, void*>& weights,
                                                       cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_norm_gamma      = weights[prefix + "norm.weight"];
    void* tmp_norm_beta       = weights[prefix + "norm.bias"];
    void* tmp_proj_in_weight  = weights[prefix + "proj_in.weight"];
    void* tmp_proj_in_bias    = weights[prefix + "proj_in.bias"];
    void* tmp_proj_out_weight = weights[prefix + "proj_out.weight"];
    void* tmp_proj_out_bias   = weights[prefix + "proj_out.bias"];

    weight_loader_manager_glob->doCudaMemcpy(norm_gamma, tmp_norm_gamma, sizeof(T) * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm_beta, tmp_norm_beta, sizeof(T) * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        proj_in_weight, tmp_proj_in_weight, sizeof(T) * inner_dim_ * 1 * 1 * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(proj_in_bias, tmp_proj_in_bias, sizeof(T) * inner_dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        proj_out_weight, tmp_proj_out_weight, sizeof(T) * in_channels_ * 1 * 1 * inner_dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * in_channels_, memcpyKind);

    if (this->quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block_weight->loadWeightsFromCache(
            prefix + std::string("transformer_blocks.0."), weights, memcpyKind);
    }
    else {
        basic_transformer_block_weight->loadWeightsFromCache(
            prefix + std::string("transformer_blocks.0."), weights, memcpyKind);
    }
}

template class Transformer2dBlockWeight<float>;
template class Transformer2dBlockWeight<half>;
}  // namespace lyradiff