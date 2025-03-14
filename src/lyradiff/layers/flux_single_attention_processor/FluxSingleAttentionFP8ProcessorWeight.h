#pragma once
#include "FluxSingleAttentionProcessorWeight.h"
#include "src/lyradiff/utils/cuda_fp8_utils.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxSingleAttentionFP8ProcessorWeight: public FluxSingleAttentionProcessorWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    __nv_fp8_e4m3* to_qkv_weight;
    T*             to_qkv_weight_h;
    T*             to_qkv_bias;
    float*         to_qkv_weight_scale;
    float*         to_qkv_input_scale;

    T* qk_norm_weight;

    FluxSingleAttentionFP8ProcessorWeight() = default;
    FluxSingleAttentionFP8ProcessorWeight(size_t        embedding_dim,
                                          size_t        embedding_head_num,
                                          size_t        embedding_head_dim,
                                          LyraQuantType quant_level,
                                          IAllocator*   allocator);

    virtual ~FluxSingleAttentionFP8ProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
