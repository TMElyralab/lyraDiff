#include "BasicTransformerInt8BlockWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <filesystem>

using namespace std;

namespace lyradiff {

template<typename T>
BasicTransformerInt8BlockWeight<T>::BasicTransformerInt8BlockWeight(const size_t        dim,
                                                                    const size_t        head_num,
                                                                    const size_t        dim_per_head,
                                                                    const size_t        cross_attn_dim,
                                                                    const LyraQuantType quant_level,
                                                                    IAllocator*         allocator):
    dim_(dim), head_num_(head_num), dim_per_head_(dim_per_head), cross_attn_dim_(cross_attn_dim)
{
    this->quant_level_ = quant_level;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    attention1_qkv_size_    = dim_ * dim_ * 3;
    attention1_to_out_size_ = dim_ * dim_;
    attention2_q_size_      = dim_ * dim_;
    attention2_kv_size_     = dim_ * cross_attn_dim_ * 2;
    attention2_to_out_size_ = dim_ * dim_;
    geglu_linear_size_      = dim_ * dim_ * 8;
    ffn_linear_size_        = dim_ * dim_ * 4;
    attention2_ip_kv_size_  = dim_ * cross_attn_dim_ * 2;
    this->allocator_        = allocator;
}

template<typename T>
BasicTransformerInt8BlockWeight<T>::~BasicTransformerInt8BlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {

        deviceFree(attention1_to_out_bias);

        deviceFree(attention2_kv_weight);

        deviceFree(attention2_to_out_bias);
        deviceFree(norm1_gamma);
        deviceFree(norm1_beta);
        deviceFree(norm2_gamma);
        deviceFree(norm2_beta);
        deviceFree(norm3_gamma);
        deviceFree(norm3_beta);
        // deviceFree(geglu_linear_weight);
        MACRODeviceFreeInt8Weights(geglu_linear);
        deviceFree(geglu_linear_bias);
        // deviceFree(ffn_linear_weight);
        MACRODeviceFreeInt8Weights(ffn_linear);
        deviceFree(ffn_linear_bias);

        if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL2
            || this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
            MACRODeviceFreeInt8Weights(attention1_qkv);
        }
        else {
            deviceFree(attention1_qkv_weight);
        }

        if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
            MACRODeviceFreeInt8Weights(attention1_to_out);
            MACRODeviceFreeInt8Weights(attention2_q);
            MACRODeviceFreeInt8Weights(attention2_to_out);
        }
        else {
            deviceFree(attention1_to_out_weight);
            deviceFree(attention2_q_weight);
            deviceFree(attention2_to_out_weight);
        }
    }

    if (is_maintain_lora) {
        deviceFree(attention1_qkv_lora_buf_);
        deviceFree(attention1_to_out_lora_buf_);
        deviceFree(attention2_q_lora_buf_);
        deviceFree(attention2_kv_lora_buf_);
        deviceFree(attention2_to_out_lora_buf_);
        deviceFree(geglu_linear_lora_buf_);
        deviceFree(ffn_linear_lora_buf_);
    }
    unLoadIPAdapter();
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        // deviceMalloc(&attention1_to_out_weight, attention1_to_out_size_);
        if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL2
            || this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
            MACRODeviceMallocInt8Weights(attention1_qkv, dim_ * 3, dim_);
        }
        else {
            deviceMalloc(&attention1_qkv_weight, attention1_qkv_size_);
        }

        deviceMalloc(&attention1_to_out_bias, dim_);

        deviceMalloc(&attention2_kv_weight, attention2_kv_size_);

        if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
            MACRODeviceMallocInt8Weights(attention1_to_out, dim_, dim_);

            MACRODeviceMallocInt8Weights(attention2_q, dim_, dim_);

            MACRODeviceMallocInt8Weights(attention2_to_out, dim_, dim_);
        }
        else {
            deviceMalloc(&attention1_to_out_weight, attention1_to_out_size_);

            deviceMalloc(&attention2_q_weight, attention2_q_size_);

            deviceMalloc(&attention2_to_out_weight, attention2_to_out_size_);
        }

        deviceMalloc(&attention2_to_out_bias, dim_);

        deviceMalloc(&norm1_gamma, dim_);
        deviceMalloc(&norm1_beta, dim_);
        deviceMalloc(&norm2_gamma, dim_);
        deviceMalloc(&norm2_beta, dim_);
        deviceMalloc(&norm3_gamma, dim_);
        deviceMalloc(&norm3_beta, dim_);

        MACRODeviceMallocInt8Weights(geglu_linear, dim_ * 8, dim_);
        deviceMalloc(&geglu_linear_bias, dim_ * 8);
        MACRODeviceMallocInt8Weights(ffn_linear, dim_, dim_ * 4);
        deviceMalloc(&ffn_linear_bias, dim_);

        // 暂时不支持lora
        // this->lora_weight_size_map = {{"attn1_to_q", dim_ * dim_},
        //                               {"attn1_to_k", dim_ * dim_},
        //                               {"attn1_to_v", dim_ * dim_},
        //                               {"attn1_to_out_0", attention1_to_out_size_},
        //                               {"attn2_to_q", attention2_q_size_},
        //                               {"attn2_to_k", dim_ * cross_attn_dim_},
        //                               {"attn2_to_v", dim_ * cross_attn_dim_},
        //                               {"attn2_to_out_0", attention2_to_out_size_},
        //                               {"ff_net_0_proj", geglu_linear_size_},
        //                               {"ff_net_2", ffn_linear_size_}};
        // this->lora_weight_map      = {{"attn1_to_q", attention1_qkv_weight},
        //                               {"attn1_to_k", &attention1_qkv_weight[dim_ * dim_]},
        //                               {"attn1_to_v", &attention1_qkv_weight[dim_ * dim_ * 2]},
        //                               {"attn1_to_out_0", attention1_to_out_weight},
        //                               {"attn2_to_q", attention2_q_weight},
        //                               {"attn2_to_k", &attention2_kv_weight[0]},
        //                               {"attn2_to_v", &attention2_kv_weight[dim_ * cross_attn_dim_]},
        //                               {"attn2_to_out_0", attention2_to_out_weight},
        //                               {"ff_net_0_proj", geglu_linear_weight},
        //                               {"ff_net_2", ffn_linear_weight}};

        is_maintain_buffer = true;
    }
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::mallocIPAdapterWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_ipadapter) {
        deviceMalloc(&attention2_ip_kv_weight, attention2_ip_kv_size_);
        is_maintain_ipadapter = true;
    }
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::unLoadIPAdapter()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_ipadapter)
        return;
    deviceFree(attention2_ip_kv_weight);
    attention2_ip_kv_weight = nullptr;
    is_maintain_ipadapter   = false;
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&attention1_qkv_lora_buf_, attention1_qkv_size_);
        deviceMalloc(&attention1_to_out_lora_buf_, attention1_to_out_size_);
        deviceMalloc(&attention2_q_lora_buf_, attention2_q_size_);
        deviceMalloc(&attention2_kv_lora_buf_, attention2_kv_size_);
        deviceMalloc(&attention2_to_out_lora_buf_, attention2_to_out_size_);
        deviceMalloc(&geglu_linear_lora_buf_, geglu_linear_size_);
        deviceMalloc(&ffn_linear_lora_buf_, ffn_linear_size_);

        is_maintain_lora = true;
    }
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                            std::string                          prefix,
                                                            std::unordered_map<std::string, T*>& lora_weights,
                                                            float                                lora_alpha,
                                                            FtCudaDataType                       lora_file_type)
{
    LYRA_CHECK_WITH_INFO(false, "BasicTransformerInt8BlockWeight::loadLoraFromWeight() is not working for now. ");
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                           std::unordered_map<std::string, T*>& lora_weights,
                                                           float                                lora_alpha,
                                                           bool                                 from_outside)
{
    LYRA_CHECK_WITH_INFO(false, "BasicTransformerInt8BlockWeight::loadLoraFromCache() is not working for now. ");
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::loadIPAdapterFromWeight(const std::string& ip_adapter_path,
                                                                 const float&       scale,
                                                                 FtCudaDataType     model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mallocIPAdapterWeights();

    std::string fpath_weight = ip_adapter_path + "/" + this->name + "attn2.to_k_ip.weight.bin";
    std::string fpath_bias   = ip_adapter_path + "/" + this->name + "attn2.to_v_ip.weight.bin";

    // ipadapter weights
    int offset = 0;
    MACROLoadKVWeightFromBin2(
        attention2_ip_kv_weight, fpath_weight, fpath_bias, dim_ * cross_attn_dim_, {dim_ * cross_attn_dim_});
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // set block name
    std::filesystem::path prefix_path = prefix;
    this->name                        = prefix_path.filename();

    mallocWeights();

    // attention1 weights
    int offset = 0;

    if (this->quant_level_ >= 2) {
        MACROLoadLinearInt8WeightFromBin(attention1_qkv, dim_ * 3, dim_, "attn1.to_qkv");
    }
    else {
        MACROLoadQKVWeightFromBin(attention1_qkv_weight, attn1, dim_ * dim_, {dim_, dim_});
    }

    if (this->quant_level_ >= 3.0) {
        MACROLoadLinearInt8WeightFromBin(attention1_to_out, dim_, dim_, "attn1.to_out.0");
        loadWeightFromBin<T>(attention1_to_out_bias, {dim_}, prefix + "attn1.to_out.0.bias.bin", model_file_type);

        MACROLoadLinearInt8WeightFromBin(attention2_q, dim_, dim_, "attn2.to_q");

        MACROLoadLinearInt8WeightFromBin(attention2_to_out, dim_, dim_, "attn2.to_out.0");
        loadWeightFromBin<T>(attention2_to_out_bias, {dim_}, prefix + "attn2.to_out.0.bias.bin", model_file_type);
    }
    else {
        MACROLoadLinerarWeightFromBin(attention1_to_out, dim_, dim_, "attn1.to_out.0");
        MACROLoadQWeightFromBin(attention2_q_weight, attn2, dim_ * dim_, {dim_, dim_});
        MACROLoadLinerarWeightFromBin(attention2_to_out, dim_, dim_, "attn2.to_out.0");
    }

    // attention2 weights
    // kv 不走int8，因为有kv cache，这里收益不大
    MACROLoadKVWeightFromBin(attention2_kv_weight, attn2, dim_ * cross_attn_dim_, {dim_, cross_attn_dim_});

    // norm weights
    MACROLoadNormWeightFromBin(norm1, dim_, "norm1");
    MACROLoadNormWeightFromBin(norm2, dim_, "norm2");
    MACROLoadNormWeightFromBin(norm3, dim_, "norm3");

    // ffn weights
    // MACROLoadLinerarWeightFromBin(geglu_linear, dim_ * 8, dim_, "ff.net.0.proj");
    // MACROLoadLinerarWeightFromBin(ffn_linear, dim_, dim_ * 4, "ff.net.2");

    MACROLoadLinearInt8WeightFromBin(geglu_linear, dim_ * 8, dim_, "ff.net.0.proj");
    loadWeightFromBin<T>(geglu_linear_bias, {dim_ * 8}, prefix + "ff.net.0.proj.bias.bin", model_file_type);

    MACROLoadLinearInt8WeightFromBin(ffn_linear, dim_, dim_ * 4, "ff.net.2");
    loadWeightFromBin<T>(ffn_linear_bias, {dim_}, prefix + "ff.net.2.bias.bin", model_file_type);
}

template<typename T>
void BasicTransformerInt8BlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                              std::unordered_map<std::string, void*>& weights,
                                                              cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // attention1 weights
    this->name = prefix;
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    int offset = 0;

    void* tmp_attn1_q           = weights[prefix + "attn1.to_q.weight"];
    void* tmp_attn1_k           = weights[prefix + "attn1.to_k.weight"];
    void* tmp_attn1_v           = weights[prefix + "attn1.to_v.weight"];
    void* tmp_attn1_to_out      = weights[prefix + "attn1.to_out.0.weight"];
    void* tmp_attn1_to_out_bias = weights[prefix + "attn1.to_out.0.bias"];

    void* tmp_attn2_q           = weights[prefix + "attn2.to_q.weight"];
    void* tmp_attn2_k           = weights[prefix + "attn2.to_k.weight"];
    void* tmp_attn2_v           = weights[prefix + "attn2.to_v.weight"];
    void* tmp_attn2_to_out      = weights[prefix + "attn2.to_out.0.weight"];
    void* tmp_attn2_to_out_bias = weights[prefix + "attn2.to_out.0.bias"];

    void* tmp_norm1_gamma = weights[prefix + "norm1.weight"];
    void* tmp_norm1_beta  = weights[prefix + "norm1.bias"];
    void* tmp_norm2_gamma = weights[prefix + "norm2.weight"];
    void* tmp_norm2_beta  = weights[prefix + "norm2.bias"];
    void* tmp_norm3_gamma = weights[prefix + "norm3.weight"];
    void* tmp_norm3_beta  = weights[prefix + "norm3.bias"];

    void* tmp_geglu_linear      = weights[prefix + "ff.net.0.proj.weight"];
    void* tmp_geglu_linear_bias = weights[prefix + "ff.net.0.proj.bias"];
    void* tmp_ffn_linear        = weights[prefix + "ff.net.2.weight"];
    void* tmp_ffn_linear_bias   = weights[prefix + "ff.net.2.bias"];

    cudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_q, sizeof(T) * dim_ * dim_, memcpyKind);
    offset += dim_ * dim_;
    cudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_k, sizeof(T) * dim_ * dim_, memcpyKind);
    offset += dim_ * dim_;
    cudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_v, sizeof(T) * dim_ * dim_, memcpyKind);

    cudaMemcpy(attention1_to_out_weight, tmp_attn1_to_out, sizeof(T) * dim_ * dim_, memcpyKind);
    cudaMemcpy(attention1_to_out_bias, tmp_attn1_to_out_bias, sizeof(T) * dim_, memcpyKind);

    cudaMemcpy(attention2_q_weight, tmp_attn2_q, sizeof(T) * attention2_q_size_, memcpyKind);

    offset = 0;
    cudaMemcpy(&attention2_kv_weight[offset], tmp_attn2_k, sizeof(T) * dim_ * cross_attn_dim_, memcpyKind);
    offset += dim_ * cross_attn_dim_;
    cudaMemcpy(&attention2_kv_weight[offset], tmp_attn2_v, sizeof(T) * dim_ * cross_attn_dim_, memcpyKind);
    cudaMemcpy(attention2_to_out_weight, tmp_attn2_to_out, sizeof(T) * dim_ * dim_, memcpyKind);
    cudaMemcpy(attention2_to_out_bias, tmp_attn2_to_out_bias, sizeof(T) * dim_, memcpyKind);

    cudaMemcpy(norm1_gamma, tmp_norm1_gamma, sizeof(T) * dim_, memcpyKind);
    cudaMemcpy(norm1_beta, tmp_norm1_beta, sizeof(T) * dim_, memcpyKind);
    cudaMemcpy(norm2_gamma, tmp_norm2_gamma, sizeof(T) * dim_, memcpyKind);
    cudaMemcpy(norm2_beta, tmp_norm2_beta, sizeof(T) * dim_, memcpyKind);
    cudaMemcpy(norm3_gamma, tmp_norm3_gamma, sizeof(T) * dim_, memcpyKind);
    cudaMemcpy(norm3_beta, tmp_norm3_beta, sizeof(T) * dim_, memcpyKind);

    cudaMemcpy(geglu_linear_weight, tmp_geglu_linear, sizeof(T) * geglu_linear_size_, memcpyKind);
    cudaMemcpy(geglu_linear_bias, tmp_geglu_linear_bias, sizeof(T) * dim_ * 8, memcpyKind);
    cudaMemcpy(ffn_linear_weight, tmp_ffn_linear, sizeof(T) * ffn_linear_size_, memcpyKind);
    cudaMemcpy(ffn_linear_bias, tmp_ffn_linear_bias, sizeof(T) * dim_, memcpyKind);
}

template class BasicTransformerInt8BlockWeight<float>;
template class BasicTransformerInt8BlockWeight<half>;
}  // namespace lyradiff