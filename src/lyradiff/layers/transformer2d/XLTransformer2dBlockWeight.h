#pragma once

#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlockWeight.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerInt8BlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {

template<typename T>
class XLTransformer2dBlockWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    size_t in_channels_;
    size_t head_num_;
    size_t dim_per_head_;
    size_t inner_dim_;
    size_t norm_num_groups_;
    size_t cross_attn_dim_;
    size_t inner_trans_num_;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    T* norm_gamma = nullptr;
    T* norm_beta  = nullptr;

    T* proj_in_weight = nullptr;
    T* proj_in_bias   = nullptr;

    T* proj_out_weight = nullptr;
    T* proj_out_bias   = nullptr;

    T* proj_in_lora_buffer  = nullptr;
    T* proj_out_lora_buffer = nullptr;

    size_t                                           proj_in_size_;
    size_t                                           proj_out_size_;
    std::vector<BasicTransformerBlockWeight<T>*>     transblock_weights;
    std::vector<BasicTransformerInt8BlockWeight<T>*> transblock_int8_weights;
    XLTransformer2dBlockWeight() = default;
    XLTransformer2dBlockWeight(size_t        in_channels,
                               size_t        head_num,
                               size_t        dim_per_head,
                               size_t        cross_attn_dim,
                               size_t        norm_num_groups,
                               size_t        inner_trans_num,
                               LyraQuantType quant_level = LyraQuantType::NONE,
                               IAllocator*   allocator   = nullptr);

    ~XLTransformer2dBlockWeight();

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
                                    FtCudaDataType                       lora_file_type,
                                    cudaStream_t                         stream);
    virtual void loadLoraFromCache(std::string                          prefix,
                                   std::unordered_map<std::string, T*>& lora_weights,
                                   float                                lora_alpha,
                                   bool                                 from_outside);
};

}  // namespace lyradiff