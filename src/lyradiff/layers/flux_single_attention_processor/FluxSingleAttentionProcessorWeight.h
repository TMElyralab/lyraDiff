#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxSingleAttentionProcessorWeight: public IFLoraWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    T* to_qkv_weight;
    T* to_qkv_weight_h;
    T* to_qkv_bias;
    T* qk_norm_weight;

    FluxSingleAttentionProcessorWeight() = default;
    FluxSingleAttentionProcessorWeight(size_t        embedding_dim,
                                       size_t        embedding_head_num,
                                       size_t        embedding_head_dim,
                                       LyraQuantType quant_level,
                                       IAllocator*   allocator);

    virtual ~FluxSingleAttentionProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
