#include "FluxSingleTransformerBlockWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxSingleTransformerBlockWeight<T>::FluxSingleTransformerBlockWeight(size_t        embedding_dim,
                                                                      size_t        embedding_head_num,
                                                                      size_t        embedding_head_dim,
                                                                      size_t        mlp_scale,
                                                                      LyraQuantType quant_level,
                                                                      IAllocator*   allocator)
{
    embedding_dim_      = embedding_dim;
    embedding_head_num_ = embedding_head_num;
    embedding_head_dim_ = embedding_head_dim;
    mlp_scale_          = mlp_scale;
    this->quant_level_  = quant_level;
    this->allocator_    = allocator;
    // cout << "cur quant level: " << this->quant_level_ << endl;
    ada_norm_weight = new AdaLayerNormWeight<T>(embedding_dim_, 3, quant_level, allocator);
    attn_weight     = new FluxSingleAttentionProcessorWeight<T>(
        embedding_dim_, embedding_head_num_, embedding_head_dim_, quant_level, allocator);

    this->lora_layer_map = {{"norm", ada_norm_weight}, {"attn", attn_weight}};

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxSingleTransformerBlockWeight<T>::~FluxSingleTransformerBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(proj_mlp_weight);
        deviceFree(proj_mlp_bias);
        deviceFree(proj_out_weight);
        deviceFree(proj_out_bias);

        if (this->quant_level_ != LyraQuantType::NONE) {
            free(proj_mlp_weight_h);
            free(proj_out_weight_h);
        }

        delete ada_norm_weight;
        delete attn_weight;
    }
}

template<typename T>
void FluxSingleTransformerBlockWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&proj_mlp_weight, mlp_scale_ * embedding_dim_ * embedding_dim_);
        deviceMalloc(&proj_mlp_bias, mlp_scale_ * embedding_dim_);
        deviceMalloc(&proj_out_weight, (mlp_scale_ + 1) * embedding_dim_ * embedding_dim_);
        deviceMalloc(&proj_out_bias, embedding_dim_);

        ada_norm_weight->mallocWeights();
        attn_weight->mallocWeights();

        if (this->quant_level_ != LyraQuantType::NONE) {
            proj_mlp_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * mlp_scale_);
            proj_out_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * (mlp_scale_ + 1));

            this->lora_weight_map = {
                {"proj_mlp",
                 new LoraWeightV2<T>(
                     {mlp_scale_ * embedding_dim_ * embedding_dim_}, proj_mlp_weight, proj_mlp_weight_h)},
                {"proj_out",
                 new LoraWeightV2<T>(
                     {(mlp_scale_ + 1) * embedding_dim_ * embedding_dim_}, proj_out_weight, proj_out_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"proj_mlp", new LoraWeight<T>({mlp_scale_ * embedding_dim_ * embedding_dim_}, proj_mlp_weight)},
                {"proj_out", new LoraWeight<T>({(mlp_scale_ + 1) * embedding_dim_ * embedding_dim_}, proj_out_weight)}};
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxSingleTransformerBlockWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeights(prefix + "norm.", model_file_type);
    attn_weight->loadWeights(prefix + "attn.", model_file_type);

    loadWeightFromBin<T>(proj_mlp_weight,
                         {mlp_scale_ * embedding_dim_ * embedding_dim_},
                         prefix + "proj_mlp.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(proj_mlp_bias, {mlp_scale_ * embedding_dim_}, prefix + "proj_mlp.bias.bin", model_file_type);

    loadWeightFromBin<T>(proj_out_weight,
                         {(mlp_scale_ + 1) * embedding_dim_ * embedding_dim_},
                         prefix + "proj_out.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(proj_out_bias, {embedding_dim_}, prefix + "proj_out.bias.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(proj_mlp_weight_h,
                        proj_mlp_weight,
                        sizeof(T) * mlp_scale_ * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(proj_out_weight_h,
                        proj_out_weight,
                        sizeof(T) * (mlp_scale_ + 1) * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template<typename T>
void FluxSingleTransformerBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                               std::unordered_map<std::string, void*>& weights,
                                                               cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeightsFromCache(prefix + "norm.", weights, memcpyKind);
    attn_weight->loadWeightsFromCache(prefix + "attn.", weights, memcpyKind);

    void* tmp_proj_mlp_weight = weights[prefix + "proj_mlp.weight"];
    void* tmp_proj_mlp_bias   = weights[prefix + "proj_mlp.bias"];

    void* tmp_proj_out_weight = weights[prefix + "proj_out.weight"];
    void* tmp_proj_out_bias   = weights[prefix + "proj_out.bias"];

    cudaMemcpy(
        proj_mlp_weight, tmp_proj_mlp_weight, sizeof(T) * mlp_scale_ * embedding_dim_ * embedding_dim_, memcpyKind);
    cudaMemcpy(proj_mlp_bias, tmp_proj_mlp_bias, sizeof(T) * mlp_scale_ * embedding_dim_, memcpyKind);
    cudaMemcpy(proj_out_weight,
               tmp_proj_out_weight,
               sizeof(T) * (mlp_scale_ + 1) * embedding_dim_ * embedding_dim_,
               memcpyKind);
    cudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * embedding_dim_, memcpyKind);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(proj_mlp_weight_h,
                        proj_mlp_weight,
                        sizeof(T) * mlp_scale_ * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(proj_out_weight_h,
                        proj_out_weight,
                        sizeof(T) * (mlp_scale_ + 1) * embedding_dim_ * embedding_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template class FluxSingleTransformerBlockWeight<float>;
template class FluxSingleTransformerBlockWeight<half>;

#ifdef ENABLE_BF16
template class FluxSingleTransformerBlockWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff