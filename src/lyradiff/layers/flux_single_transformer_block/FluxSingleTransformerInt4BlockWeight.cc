#include "FluxSingleTransformerInt4BlockWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxSingleTransformerInt4BlockWeight<T>::FluxSingleTransformerInt4BlockWeight(size_t        embedding_dim,
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
    // quant_level = LyraQuantType::FP8_W8A8;
    this->quant_level_ = quant_level;
    this->allocator_   = allocator;
    if (quant_level == LyraQuantType::INT4_W4A4_FULL) {
        ada_norm_weight = new AdaFP8LayerNormWeight<T>(embedding_dim_, 3, quant_level, allocator);
    }
    else {
        ada_norm_weight = new AdaLayerNormWeight<T>(embedding_dim_, 3, quant_level, allocator);
    }
    attn_weight = new FluxSingleAttentionInt4ProcessorWeight<T>(
        embedding_dim_, embedding_head_num_, embedding_head_dim_, quant_level, allocator);

    // this->lora_layer_map = {{"norm", ada_norm_weight}, {"attn", attn_weight}};

    proj_mlp_gemm_weight   = new W4A4GemmWeight<T>(mlp_scale_ * embedding_dim_, embedding_dim_, 32, true);
    proj_out_1_gemm_weight = new W4A4GemmWeight<T>(embedding_dim_, embedding_dim_, 32, true);
    proj_out_2_gemm_weight = new W4A4GemmWeight<T>(embedding_dim_, embedding_dim_ * mlp_scale_, 32, false);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxSingleTransformerInt4BlockWeight<T>::~FluxSingleTransformerInt4BlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {

        delete ada_norm_weight;
        delete attn_weight;
        delete proj_mlp_gemm_weight;
        delete proj_out_1_gemm_weight;
        delete proj_out_2_gemm_weight;
    }
}

template<typename T>
void FluxSingleTransformerInt4BlockWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        ada_norm_weight->mallocWeights();
        attn_weight->mallocWeights();
        proj_mlp_gemm_weight->mallocWeights();
        proj_out_1_gemm_weight->mallocWeights();
        proj_out_2_gemm_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxSingleTransformerInt4BlockWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeights(prefix + "norm.", model_file_type);
    attn_weight->loadWeights(prefix + "attn.", model_file_type);
    proj_mlp_gemm_weight->loadWeights(prefix + "proj_mlp.", model_file_type);
    proj_out_1_gemm_weight->loadWeights(prefix + "proj_out.linears.0.", model_file_type);
    proj_out_2_gemm_weight->loadWeights(prefix + "proj_out.linears.1.linear.", model_file_type);
}

template<typename T>
void FluxSingleTransformerInt4BlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                                   std::unordered_map<std::string, void*>& weights,
                                                                   cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    ada_norm_weight->loadWeightsFromCache(prefix + "norm.", weights, memcpyKind);
    attn_weight->loadWeightsFromCache(prefix + "attn.", weights, memcpyKind);

    // size_t cur_size     = 0;
    // size_t cur_size_2   = 0;
    // size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * (mlp_scale_ + 1);
    // T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size,
    // false);

    // cur_size   = embedding_dim_ * embedding_dim_ * mlp_scale_;
    // cur_size_2 = embedding_dim_ * mlp_scale_;
    // MACROLoadFP8WeightFromCache(proj_mlp, cur_size, cur_size_2, "proj_mlp");

    // cur_size   = embedding_dim_ * embedding_dim_ * (mlp_scale_ + 1);
    // cur_size_2 = embedding_dim_;
    // MACROLoadFP8WeightFromCache(proj_out, cur_size, cur_size_2, "proj_out")

    // cudaMemcpy(proj_out_weight,
    //            tmp_proj_out_weight,
    //            sizeof(T) * (mlp_scale_ + 1) * embedding_dim_ * embedding_dim_,
    //            memcpyKind);
    // cudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * embedding_dim_, memcpyKind);
}

template class FluxSingleTransformerInt4BlockWeight<float>;
template class FluxSingleTransformerInt4BlockWeight<half>;

#ifdef ENABLE_BF16
template class FluxSingleTransformerInt4BlockWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff