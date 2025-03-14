#pragma once
#include "FluxSingleAttentionProcessorWeight.h"
#include "src/lyradiff/layers/w4a4_gemm_block/W4A4GemmWeight.h"
#include "src/lyradiff/utils/cuda_fp8_utils.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxSingleAttentionInt4ProcessorWeight: public FluxSingleAttentionProcessorWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    T* qk_norm_weight;

    W4A4GemmWeight<T>* to_qkv_weight;

    FluxSingleAttentionInt4ProcessorWeight() = default;
    FluxSingleAttentionInt4ProcessorWeight(size_t        embedding_dim,
                                           size_t        embedding_head_num,
                                           size_t        embedding_head_dim,
                                           LyraQuantType quant_level,
                                           IAllocator*   allocator);

    virtual ~FluxSingleAttentionInt4ProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
