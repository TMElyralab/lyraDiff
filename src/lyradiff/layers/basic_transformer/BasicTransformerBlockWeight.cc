#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlockWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"
#include <filesystem>

using namespace std;

namespace lyradiff {

template<typename T>
BasicTransformerBlockWeight<T>::BasicTransformerBlockWeight(const size_t dim,
                                                            const size_t head_num,
                                                            const size_t dim_per_head,
                                                            const size_t cross_attn_dim,
                                                            IAllocator*  allocator):
    dim_(dim), head_num_(head_num), dim_per_head_(dim_per_head), cross_attn_dim_(cross_attn_dim)
{
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
BasicTransformerBlockWeight<T>::~BasicTransformerBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(attention1_qkv_weight);
        deviceFree(attention1_to_out_weight);
        deviceFree(attention1_to_out_bias);
        deviceFree(attention2_q_weight);
        deviceFree(attention2_kv_weight);
        deviceFree(attention2_to_out_weight);
        deviceFree(attention2_to_out_bias);
        deviceFree(norm1_gamma);
        deviceFree(norm1_beta);
        deviceFree(norm2_gamma);
        deviceFree(norm2_beta);
        deviceFree(norm3_gamma);
        deviceFree(norm3_beta);
        deviceFree(geglu_linear_weight);
        deviceFree(geglu_linear_bias);
        deviceFree(ffn_linear_weight);
        deviceFree(ffn_linear_bias);

        attention1_qkv_weight    = nullptr;
        attention1_to_out_weight = nullptr;
        attention1_to_out_bias   = nullptr;

        attention2_q_weight      = nullptr;
        attention2_kv_weight     = nullptr;
        attention2_to_out_weight = nullptr;
        attention2_to_out_bias   = nullptr;

        norm1_gamma = nullptr;
        norm1_beta  = nullptr;

        norm2_gamma = nullptr;
        norm2_beta  = nullptr;

        norm3_gamma = nullptr;
        norm3_beta  = nullptr;

        geglu_linear_weight = nullptr;
        geglu_linear_bias   = nullptr;

        ffn_linear_weight = nullptr;
        ffn_linear_bias   = nullptr;
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
void BasicTransformerBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        deviceMalloc(&attention1_qkv_weight, attention1_qkv_size_);
        deviceMalloc(&attention1_to_out_weight, attention1_to_out_size_);
        deviceMalloc(&attention1_to_out_bias, dim_);

        deviceMalloc(&attention2_q_weight, attention2_q_size_);
        deviceMalloc(&attention2_kv_weight, attention2_kv_size_);
        deviceMalloc(&attention2_to_out_weight, attention2_to_out_size_);
        deviceMalloc(&attention2_to_out_bias, dim_);

        deviceMalloc(&norm1_gamma, dim_);
        deviceMalloc(&norm1_beta, dim_);
        deviceMalloc(&norm2_gamma, dim_);
        deviceMalloc(&norm2_beta, dim_);
        deviceMalloc(&norm3_gamma, dim_);
        deviceMalloc(&norm3_beta, dim_);

        deviceMalloc(&geglu_linear_weight, geglu_linear_size_);
        deviceMalloc(&geglu_linear_bias, dim_ * 8);
        deviceMalloc(&ffn_linear_weight, ffn_linear_size_);
        deviceMalloc(&ffn_linear_bias, dim_);

        this->lora_weight_map = {
            {"attn1_to_q", new LoraWeight<T>({dim_, dim_}, attention1_qkv_weight)},
            {"attn1_to_k", new LoraWeight<T>({dim_, dim_}, &attention1_qkv_weight[dim_ * dim_])},
            {"attn1_to_v", new LoraWeight<T>({dim_, dim_}, &attention1_qkv_weight[dim_ * dim_ * 2])},
            {"attn1_to_qkv", new LoraWeight<T>({dim_ * 3, dim_}, attention1_qkv_weight)},
            {"attn1_to_out_0", new LoraWeight<T>({dim_, dim_}, attention1_to_out_weight)},
            {"attn2_to_q", new LoraWeight<T>({dim_, dim_}, attention2_q_weight)},
            {"attn2_to_k", new LoraWeight<T>({dim_, cross_attn_dim_}, attention2_kv_weight)},
            {"attn2_to_v", new LoraWeight<T>({dim_, cross_attn_dim_}, &attention2_kv_weight[dim_ * cross_attn_dim_])},
            {"attn2_to_out_0", new LoraWeight<T>({dim_, dim_}, attention2_to_out_weight)},
            {"ff_net_0_proj", new LoraWeight<T>({dim_ * 8, dim_}, geglu_linear_weight)},
            {"ff_net_2", new LoraWeight<T>({dim_, dim_ * 4}, ffn_linear_weight)}};

        is_maintain_buffer = true;
    }
}

template<typename T>
void BasicTransformerBlockWeight<T>::mallocIPAdapterWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_ipadapter) {
        deviceMalloc(&attention2_ip_kv_weight, attention2_ip_kv_size_);
        is_maintain_ipadapter = true;
    }
}

template<typename T>
void BasicTransformerBlockWeight<T>::unLoadIPAdapter()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_ipadapter)
        return;
    deviceFree(attention2_ip_kv_weight);
    attention2_ip_kv_weight = nullptr;
    is_maintain_ipadapter   = false;
}

template<typename T>
void BasicTransformerBlockWeight<T>::mallocLoraBuffer()
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
void BasicTransformerBlockWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                        std::string                          prefix,
                                                        std::unordered_map<std::string, T*>& lora_weights,
                                                        float                                lora_alpha,
                                                        FtCudaDataType                       lora_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }

    int offset = 0;
    loadWeightFromBin<T>(
        &attention1_qkv_lora_buf_[offset], {dim_, dim_}, lora_path + prefix + "attn1_to_q.bin", lora_file_type);
    offset += dim_ * dim_;
    loadWeightFromBin<T>(
        &attention1_qkv_lora_buf_[offset], {dim_, dim_}, lora_path + prefix + "attn1_to_k.bin", lora_file_type);
    offset += dim_ * dim_;
    loadWeightFromBin<T>(
        &attention1_qkv_lora_buf_[offset], {dim_, dim_}, lora_path + prefix + "attn1_to_v.bin", lora_file_type);
    loadWeightFromBin<T>(
        attention1_to_out_lora_buf_, {dim_, dim_}, lora_path + prefix + "attn1_to_out_0.bin", lora_file_type);
    // attention2 weights
    loadWeightFromBin<T>(attention2_q_lora_buf_, {dim_, dim_}, lora_path + prefix + "attn2_to_q.bin", lora_file_type);
    offset = 0;
    loadWeightFromBin<T>(&attention2_kv_lora_buf_[offset],
                         {dim_, cross_attn_dim_},
                         lora_path + prefix + "attn2_to_k.bin",
                         lora_file_type);
    offset += dim_ * cross_attn_dim_;
    loadWeightFromBin<T>(&attention2_kv_lora_buf_[offset],
                         {dim_, cross_attn_dim_},
                         lora_path + prefix + "attn2_to_v.bin",
                         lora_file_type);

    loadWeightFromBin<T>(
        attention2_to_out_lora_buf_, {dim_, dim_}, lora_path + prefix + "attn2_to_out_0.bin", lora_file_type);

    loadWeightFromBin<T>(
        geglu_linear_lora_buf_, {dim_ * 8, dim_}, lora_path + prefix + "ff_net_0_proj.bin", lora_file_type);
    loadWeightFromBin<T>(ffn_linear_lora_buf_, {dim_, dim_ * 4}, lora_path + prefix + "ff_net_2.bin", lora_file_type);

    T* attn1_qkv    = (T*)malloc(sizeof(T) * attention1_qkv_size_);
    T* attn1_to_out = (T*)malloc(sizeof(T) * attention1_to_out_size_);
    T* attn2_q      = (T*)malloc(sizeof(T) * attention2_q_size_);
    T* attn2_kv     = (T*)malloc(sizeof(T) * attention2_kv_size_);
    T* attn2_to_out = (T*)malloc(sizeof(T) * attention2_to_out_size_);
    T* geglu_linear = (T*)malloc(sizeof(T) * geglu_linear_size_);
    T* ffn_linear   = (T*)malloc(sizeof(T) * ffn_linear_size_);

    // 再缓存到本地
    cudaMemcpy(attn1_qkv, attention1_qkv_lora_buf_, sizeof(T) * attention1_qkv_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(attn1_to_out, attention1_to_out_lora_buf_, sizeof(T) * attention1_to_out_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(attn2_q, attention2_q_lora_buf_, sizeof(T) * attention2_q_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(attn2_kv, attention2_kv_lora_buf_, sizeof(T) * attention2_kv_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(attn2_to_out, attention2_to_out_lora_buf_, sizeof(T) * attention2_to_out_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(geglu_linear, geglu_linear_lora_buf_, sizeof(T) * geglu_linear_size_, cudaMemcpyDeviceToHost);
    cudaMemcpy(ffn_linear, ffn_linear_lora_buf_, sizeof(T) * ffn_linear_size_, cudaMemcpyDeviceToHost);

    lora_weights[prefix + "attn1_qkv"]    = attn1_qkv;
    lora_weights[prefix + "attn1_to_out"] = attn1_to_out;
    lora_weights[prefix + "attn2_q"]      = attn2_q;
    lora_weights[prefix + "attn2_kv"]     = attn2_kv;
    lora_weights[prefix + "attn2_to_out"] = attn2_to_out;
    lora_weights[prefix + "geglu_linear"] = geglu_linear;
    lora_weights[prefix + "ffn_linear"]   = ffn_linear;

    invokeLoadLora<T>(attention1_qkv_weight, attention1_qkv_lora_buf_, attention1_qkv_size_, lora_alpha);
    invokeLoadLora<T>(attention1_to_out_weight, attention1_to_out_lora_buf_, attention1_to_out_size_, lora_alpha);
    invokeLoadLora<T>(attention2_q_weight, attention2_q_lora_buf_, attention2_q_size_, lora_alpha);
    invokeLoadLora<T>(attention2_kv_weight, attention2_kv_lora_buf_, attention2_kv_size_, lora_alpha);
    invokeLoadLora<T>(attention2_to_out_weight, attention2_to_out_lora_buf_, attention2_to_out_size_, lora_alpha);
    invokeLoadLora<T>(geglu_linear_weight, geglu_linear_lora_buf_, geglu_linear_size_, lora_alpha);
    invokeLoadLora<T>(ffn_linear_weight, ffn_linear_lora_buf_, ffn_linear_size_, lora_alpha);
}

template<typename T>
void BasicTransformerBlockWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                       std::unordered_map<std::string, T*>& lora_weights,
                                                       float                                lora_alpha,
                                                       bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }

    if (from_outside) {
        T* attn1_q      = lora_weights[prefix + "attn1_to_q"];
        T* attn1_k      = lora_weights[prefix + "attn1_to_k"];
        T* attn1_v      = lora_weights[prefix + "attn1_to_v"];
        T* attn1_to_out = lora_weights[prefix + "attn1_to_out_0"];
        T* attn2_q      = lora_weights[prefix + "attn2_to_q"];
        T* attn2_k      = lora_weights[prefix + "attn2_to_k"];
        T* attn2_v      = lora_weights[prefix + "attn2_to_v"];
        T* attn2_to_out = lora_weights[prefix + "attn2_to_out_0"];
        T* geglu_linear = lora_weights[prefix + "ff_net_0_proj"];
        T* ffn_linear   = lora_weights[prefix + "ff_net_2"];

        int offset = 0;
        cudaMemcpy(&attention1_qkv_lora_buf_[offset], attn1_q, sizeof(T) * dim_ * dim_, cudaMemcpyHostToDevice);
        offset += dim_ * dim_;
        cudaMemcpy(&attention1_qkv_lora_buf_[offset], attn1_k, sizeof(T) * dim_ * dim_, cudaMemcpyHostToDevice);
        offset += dim_ * dim_;
        cudaMemcpy(&attention1_qkv_lora_buf_[offset], attn1_v, sizeof(T) * dim_ * dim_, cudaMemcpyHostToDevice);

        cudaMemcpy(
            attention1_to_out_lora_buf_, attn1_to_out, sizeof(T) * attention1_to_out_size_, cudaMemcpyHostToDevice);

        cudaMemcpy(attention2_q_lora_buf_, attn2_q, sizeof(T) * attention2_q_size_, cudaMemcpyHostToDevice);

        offset = 0;
        cudaMemcpy(
            &attention2_kv_lora_buf_[offset], attn2_k, sizeof(T) * dim_ * cross_attn_dim_, cudaMemcpyHostToDevice);
        offset += dim_ * cross_attn_dim_;
        cudaMemcpy(
            &attention2_kv_lora_buf_[offset], attn2_v, sizeof(T) * dim_ * cross_attn_dim_, cudaMemcpyHostToDevice);

        cudaMemcpy(
            attention2_to_out_lora_buf_, attn2_to_out, sizeof(T) * attention2_to_out_size_, cudaMemcpyHostToDevice);
        cudaMemcpy(geglu_linear_lora_buf_, geglu_linear, sizeof(T) * geglu_linear_size_, cudaMemcpyHostToDevice);
        cudaMemcpy(ffn_linear_lora_buf_, ffn_linear, sizeof(T) * ffn_linear_size_, cudaMemcpyHostToDevice);

        invokeLoadLora<T>(attention1_qkv_weight, attention1_qkv_lora_buf_, attention1_qkv_size_, lora_alpha);
        invokeLoadLora<T>(attention1_to_out_weight, attention1_to_out_lora_buf_, attention1_to_out_size_, lora_alpha);
        invokeLoadLora<T>(attention2_q_weight, attention2_q_lora_buf_, attention2_q_size_, lora_alpha);
        invokeLoadLora<T>(attention2_kv_weight, attention2_kv_lora_buf_, attention2_kv_size_, lora_alpha);
        invokeLoadLora<T>(attention2_to_out_weight, attention2_to_out_lora_buf_, attention2_to_out_size_, lora_alpha);
        invokeLoadLora<T>(geglu_linear_weight, geglu_linear_lora_buf_, geglu_linear_size_, lora_alpha);
        invokeLoadLora<T>(ffn_linear_weight, ffn_linear_lora_buf_, ffn_linear_size_, lora_alpha);

        return;
    }

    T* attn1_qkv    = lora_weights[prefix + "attn1_qkv"];
    T* attn1_to_out = lora_weights[prefix + "attn1_to_out"];
    T* attn2_q      = lora_weights[prefix + "attn2_q"];
    T* attn2_kv     = lora_weights[prefix + "attn2_kv"];
    T* attn2_to_out = lora_weights[prefix + "attn2_to_out"];
    T* geglu_linear = lora_weights[prefix + "geglu_linear"];
    T* ffn_linear   = lora_weights[prefix + "ffn_linear"];

    cudaMemcpy(attention1_qkv_lora_buf_, attn1_qkv, sizeof(T) * attention1_qkv_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(attention1_to_out_lora_buf_, attn1_to_out, sizeof(T) * attention1_to_out_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(attention2_q_lora_buf_, attn2_q, sizeof(T) * attention2_q_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(attention2_kv_lora_buf_, attn2_kv, sizeof(T) * attention2_kv_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(attention2_to_out_lora_buf_, attn2_to_out, sizeof(T) * attention2_to_out_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(geglu_linear_lora_buf_, geglu_linear, sizeof(T) * geglu_linear_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(ffn_linear_lora_buf_, ffn_linear, sizeof(T) * ffn_linear_size_, cudaMemcpyHostToDevice);

    invokeLoadLora<T>(attention1_qkv_weight, attention1_qkv_lora_buf_, attention1_qkv_size_, lora_alpha);
    invokeLoadLora<T>(attention1_to_out_weight, attention1_to_out_lora_buf_, attention1_to_out_size_, lora_alpha);
    invokeLoadLora<T>(attention2_q_weight, attention2_q_lora_buf_, attention2_q_size_, lora_alpha);
    invokeLoadLora<T>(attention2_kv_weight, attention2_kv_lora_buf_, attention2_kv_size_, lora_alpha);
    invokeLoadLora<T>(attention2_to_out_weight, attention2_to_out_lora_buf_, attention2_to_out_size_, lora_alpha);
    invokeLoadLora<T>(geglu_linear_weight, geglu_linear_lora_buf_, geglu_linear_size_, lora_alpha);
    invokeLoadLora<T>(ffn_linear_weight, ffn_linear_lora_buf_, ffn_linear_size_, lora_alpha);
}

template<typename T>
void BasicTransformerBlockWeight<T>::loadIPAdapterFromWeight(const std::string& ip_adapter_path,
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
void BasicTransformerBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // set block name
    std::filesystem::path prefix_path = prefix;
    this->name                        = prefix_path.filename();

    mallocWeights();

    // attention1 weights
    int offset = 0;
    MACROLoadQKVWeightFromBin(attention1_qkv_weight, attn1, dim_ * dim_, {dim_, dim_});
    MACROLoadLinerarWeightFromBin(attention1_to_out, dim_, dim_, "attn1.to_out.0");

    // attention2 weights
    MACROLoadQWeightFromBin(attention2_q_weight, attn2, dim_ * dim_, {dim_, dim_});
    MACROLoadKVWeightFromBin(attention2_kv_weight, attn2, dim_ * cross_attn_dim_, {dim_, cross_attn_dim_});
    MACROLoadLinerarWeightFromBin(attention2_to_out, dim_, dim_, "attn2.to_out.0");

    // norm weights
    MACROLoadNormWeightFromBin(norm1, dim_, "norm1");
    MACROLoadNormWeightFromBin(norm2, dim_, "norm2");
    MACROLoadNormWeightFromBin(norm3, dim_, "norm3");

    // ffn weights
    MACROLoadLinerarWeightFromBin(geglu_linear, dim_ * 8, dim_, "ff.net.0.proj");
    MACROLoadLinerarWeightFromBin(ffn_linear, dim_, dim_ * 4, "ff.net.2");
}

template<typename T>
void BasicTransformerBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
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

    // printf("weight_loader_manager_glob revmap sz: %d\n", weight_loader_manager_glob->map_weights_reverse.size());
    // printf("name debug: %s\n", weight_loader_manager_glob->map_weights_reverse[tmp_attn1_q].c_str());

    weight_loader_manager_glob->doCudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_q, sizeof(T) * dim_ * dim_, memcpyKind);
    offset += dim_ * dim_;
    weight_loader_manager_glob->doCudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_k, sizeof(T) * dim_ * dim_, memcpyKind);
    offset += dim_ * dim_;
    weight_loader_manager_glob->doCudaMemcpy(&attention1_qkv_weight[offset], tmp_attn1_v, sizeof(T) * dim_ * dim_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(attention1_to_out_weight, tmp_attn1_to_out, sizeof(T) * dim_ * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(attention1_to_out_bias, tmp_attn1_to_out_bias, sizeof(T) * dim_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(attention2_q_weight, tmp_attn2_q, sizeof(T) * attention2_q_size_, memcpyKind);

    offset = 0;
    weight_loader_manager_glob->doCudaMemcpy(&attention2_kv_weight[offset], tmp_attn2_k, sizeof(T) * dim_ * cross_attn_dim_, memcpyKind);
    offset += dim_ * cross_attn_dim_;
    weight_loader_manager_glob->doCudaMemcpy(&attention2_kv_weight[offset], tmp_attn2_v, sizeof(T) * dim_ * cross_attn_dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(attention2_to_out_weight, tmp_attn2_to_out, sizeof(T) * dim_ * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(attention2_to_out_bias, tmp_attn2_to_out_bias, sizeof(T) * dim_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(norm1_gamma, tmp_norm1_gamma, sizeof(T) * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm1_beta, tmp_norm1_beta, sizeof(T) * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm2_gamma, tmp_norm2_gamma, sizeof(T) * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm2_beta, tmp_norm2_beta, sizeof(T) * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm3_gamma, tmp_norm3_gamma, sizeof(T) * dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(norm3_beta, tmp_norm3_beta, sizeof(T) * dim_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(geglu_linear_weight, tmp_geglu_linear, sizeof(T) * geglu_linear_size_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(geglu_linear_bias, tmp_geglu_linear_bias, sizeof(T) * dim_ * 8, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(ffn_linear_weight, tmp_ffn_linear, sizeof(T) * ffn_linear_size_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(ffn_linear_bias, tmp_ffn_linear_bias, sizeof(T) * dim_, memcpyKind);
}

template class BasicTransformerBlockWeight<float>;
template class BasicTransformerBlockWeight<half>;
}  // namespace lyradiff