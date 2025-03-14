#pragma once

#include "src/lyradiff/interface/IFCBase.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <unordered_map>
namespace lyradiff {

template<typename T>
class BasicTransformerBlockWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    size_t dim_;
    size_t head_num_;
    size_t dim_per_head_;
    size_t cross_attn_dim_;

    size_t attention1_qkv_size_;
    size_t attention1_to_out_size_;
    size_t attention2_q_size_;
    size_t attention2_kv_size_;
    size_t attention2_to_out_size_;
    size_t geglu_linear_size_;
    size_t ffn_linear_size_;

    // ip_adapter
    size_t attention2_ip_kv_size_;

protected:
    bool is_maintain_buffer    = false;
    bool is_maintain_lora      = false;
    bool is_maintain_ipadapter = false;
    bool is_maintain_fp        = false;

public:
    std::string name;
    T*          attention1_qkv_weight    = nullptr;
    T*          attention1_to_out_weight = nullptr;
    T*          attention1_to_out_bias   = nullptr;

    T* attention2_q_weight      = nullptr;
    T* attention2_kv_weight     = nullptr;
    T* attention2_to_out_weight = nullptr;
    T* attention2_to_out_bias   = nullptr;

    T* attention2_ip_kv_weight = nullptr;
    T* attention2_fp_kv_weight = nullptr;

    T* norm1_gamma = nullptr;
    T* norm1_beta  = nullptr;

    T* norm2_gamma = nullptr;
    T* norm2_beta  = nullptr;

    T* norm3_gamma = nullptr;
    T* norm3_beta  = nullptr;

    T* geglu_linear_weight = nullptr;
    T* geglu_linear_bias   = nullptr;

    T* ffn_linear_weight = nullptr;
    T* ffn_linear_bias   = nullptr;

    T* attention1_qkv_lora_buf_    = nullptr;
    T* attention1_to_out_lora_buf_ = nullptr;
    T* attention2_q_lora_buf_      = nullptr;
    T* attention2_kv_lora_buf_     = nullptr;
    T* attention2_to_out_lora_buf_ = nullptr;
    T* geglu_linear_lora_buf_      = nullptr;
    T* ffn_linear_lora_buf_        = nullptr;

    BasicTransformerBlockWeight() = default;
    BasicTransformerBlockWeight(const size_t dim,
                                const size_t head_num,
                                const size_t dim_per_head,
                                const size_t cross_attn_dim,
                                IAllocator*  allocator = nullptr);

    ~BasicTransformerBlockWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
    virtual void mallocLoraBuffer();
    virtual void loadLoraFromWeight(std::string                          lora_path,
                                    std::string                          prefix,
                                    std::unordered_map<std::string, T*>& lora_weights,
                                    float                                lora_alpha,
                                    FtCudaDataType                       lora_file_type);
    virtual void loadLoraFromCache(std::string                          prefix,
                                   std::unordered_map<std::string, T*>& lora_weights,
                                   float                                lora_alpha,
                                   bool                                 from_outside = true);
    virtual void mallocIPAdapterWeights();
    virtual void
    loadIPAdapterFromWeight(const std::string& ip_adapter_path, const float& scale, FtCudaDataType model_file_type);
    virtual void unLoadIPAdapter();
    virtual bool hasIPAdapter() const
    {
        return is_maintain_ipadapter;
    }
};

}  // namespace lyradiff
