#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlockWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLTransformer2dBlockWeight<T>::XLTransformer2dBlockWeight(size_t        in_channels,
                                                          size_t        head_num,
                                                          size_t        dim_per_head,
                                                          size_t        cross_attn_dim,
                                                          size_t        norm_num_groups,
                                                          size_t        inner_trans_num,
                                                          LyraQuantType quant_level,
                                                          IAllocator*   allocator):
    in_channels_(in_channels),
    head_num_(head_num),
    dim_per_head_(dim_per_head),
    inner_dim_(head_num * dim_per_head),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim),
    inner_trans_num_(inner_trans_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    proj_in_size_  = inner_dim_ * 1 * 1 * in_channels_;
    proj_out_size_ = in_channels_ * 1 * 1 * inner_dim_;

    this->quant_level_ = quant_level;
    this->allocator_   = allocator;
    if (this->quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < inner_trans_num_; i++) {
            BasicTransformerInt8BlockWeight<T>* tmp = new BasicTransformerInt8BlockWeight<T>(
                inner_dim_, head_num_, dim_per_head_, cross_attn_dim_, quant_level, allocator);
            transblock_int8_weights.emplace_back(tmp);
            this->vec_basic_transformer_container_weights.emplace_back(tmp);

            std::string cur_layer_name           = "transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }
    }
    else {
        for (int i = 0; i < inner_trans_num_; i++) {
            BasicTransformerBlockWeight<T>* tmp =
                new BasicTransformerBlockWeight<T>(inner_dim_, head_num_, dim_per_head_, cross_attn_dim_, allocator);
            transblock_weights.emplace_back(tmp);
            this->vec_basic_transformer_container_weights.emplace_back(tmp);

            std::string cur_layer_name           = "transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }
    }
}

template<typename T>
XLTransformer2dBlockWeight<T>::~XLTransformer2dBlockWeight()
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
            for (int i = 0; i < transblock_int8_weights.size(); i++) {
                delete transblock_int8_weights[i];
                transblock_int8_weights[i] = nullptr;
            }
        }
        else {
            for (int i = 0; i < transblock_weights.size(); i++) {
                delete transblock_weights[i];
                transblock_weights[i] = nullptr;
            }
        }
    }

    if (is_maintain_lora) {
        deviceFree(proj_in_lora_buffer);
        deviceFree(proj_out_lora_buffer);
    }
}

template<typename T>
void XLTransformer2dBlockWeight<T>::mallocWeights()
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

        this->lora_weight_map = {{"proj_in", new LoraWeight<T>({inner_dim_, in_channels_}, proj_in_weight)},
                                 {"proj_out", new LoraWeight<T>({in_channels_, inner_dim_}, proj_out_weight)}};

        if (this->quant_level_ != LyraQuantType::NONE) {
            for (int i = 0; i < transblock_int8_weights.size(); i++) {
                transblock_int8_weights[i]->mallocWeights();
            }
        }
        else {
            for (int i = 0; i < transblock_weights.size(); i++) {
                transblock_weights[i]->mallocWeights();
            }
        }

        // mallocLoraBuffer();
    }
    is_maintain_buffer = true;
}

template<typename T>
void XLTransformer2dBlockWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_lora) {
        deviceMalloc(&proj_in_lora_buffer, proj_in_size_);
        deviceMalloc(&proj_out_lora_buffer, proj_out_size_);
    }
    is_maintain_lora = true;
}

template<typename T>
void XLTransformer2dBlockWeight<T>::loadLoraFromWeight(std::string                          lora_path,
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
        for (int i = 0; i < transblock_int8_weights.size(); i++) {
            transblock_int8_weights[i]->loadLoraFromWeight(lora_path,
                                                           prefix + "transformer_blocks_" + to_string(i)
                                                               + std::string("_"),
                                                           lora_weights,
                                                           lora_alpha,
                                                           lora_file_type);
        }
    }
    else {
        for (int i = 0; i < transblock_weights.size(); i++) {
            transblock_weights[i]->loadLoraFromWeight(lora_path,
                                                      prefix + "transformer_blocks_" + to_string(i) + std::string("_"),
                                                      lora_weights,
                                                      lora_alpha,
                                                      lora_file_type);
        }
    }
}

template<typename T>
void XLTransformer2dBlockWeight<T>::loadLoraFromCache(std::string                          prefix,
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

    if (this->quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < transblock_int8_weights.size(); i++) {
            transblock_int8_weights[i]->loadLoraFromCache(prefix + "transformer_blocks_" + to_string(i)
                                                              + std::string("_"),
                                                          lora_weights,
                                                          lora_alpha,
                                                          from_outside);
        }
    }
    else {
        for (int i = 0; i < transblock_weights.size(); i++) {
            transblock_weights[i]->loadLoraFromCache(prefix + "transformer_blocks_" + to_string(i) + std::string("_"),
                                                     lora_weights,
                                                     lora_alpha,
                                                     from_outside);
        }
    }
}

template<typename T>
void XLTransformer2dBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
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
        for (int i = 0; i < transblock_int8_weights.size(); i++) {
            transblock_int8_weights[i]->loadWeights(
                prefix + std::string("transformer_blocks.") + to_string(i) + std::string("."), model_file_type);
        }
    }
    else {
        for (int i = 0; i < transblock_weights.size(); i++) {
            transblock_weights[i]->loadWeights(
                prefix + std::string("transformer_blocks.") + to_string(i) + std::string("."), model_file_type);
        }
    }
}

template<typename T>
void XLTransformer2dBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
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

    cudaMemcpy(norm_gamma, tmp_norm_gamma, sizeof(T) * in_channels_, memcpyKind);
    cudaMemcpy(norm_beta, tmp_norm_beta, sizeof(T) * in_channels_, memcpyKind);
    cudaMemcpy(proj_in_weight, tmp_proj_in_weight, sizeof(T) * inner_dim_ * 1 * 1 * in_channels_, memcpyKind);
    cudaMemcpy(proj_in_bias, tmp_proj_in_bias, sizeof(T) * inner_dim_, memcpyKind);
    cudaMemcpy(proj_out_weight, tmp_proj_out_weight, sizeof(T) * in_channels_ * 1 * 1 * inner_dim_, memcpyKind);
    cudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * in_channels_, memcpyKind);

    if (this->quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < transblock_int8_weights.size(); i++) {
            transblock_int8_weights[i]->loadWeightsFromCache(
                prefix + std::string("transformer_blocks.") + to_string(i) + std::string("."), weights, memcpyKind);
        }
    }
    else {
        for (int i = 0; i < transblock_weights.size(); i++) {
            transblock_weights[i]->loadWeightsFromCache(
                prefix + std::string("transformer_blocks.") + to_string(i) + std::string("."), weights, memcpyKind);
        }
    }
}

template class XLTransformer2dBlockWeight<float>;
template class XLTransformer2dBlockWeight<half>;
}  // namespace lyradiff