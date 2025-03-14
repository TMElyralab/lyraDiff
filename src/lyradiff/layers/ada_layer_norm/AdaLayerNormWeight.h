#pragma once
#include "src/lyradiff/utils/cuda_utils.h"
#include <src/lyradiff/interface/IFLoraWeight.h>

namespace lyradiff {

template<typename T>
class AdaLayerNormWeight: public IFLoraWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_scale_;

protected:
    bool is_maintain_buffer = false;

public:
    T* linear_weight;
    T* linear_bias;

    T* linear_weight_h;

    AdaLayerNormWeight() = default;
    AdaLayerNormWeight(size_t        embedding_dim,
                       size_t        embedding_scale,
                       LyraQuantType quant_level,
                       IAllocator*   allocator = nullptr);

    virtual ~AdaLayerNormWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
