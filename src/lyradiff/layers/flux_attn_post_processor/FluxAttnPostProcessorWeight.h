#pragma once
#include "src/lyradiff/utils/cuda_utils.h"
#include <src/lyradiff/interface/IFLoraWeight.h>

namespace lyradiff {

template<typename T>
class FluxAttnPostProcessorWeight: public IFLoraWeight<T> {
private:
    size_t embedding_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    T* gelu_proj_weight;
    T* gelu_proj_weight_h;
    T* gelu_proj_bias;

    T* ff_linear_weight;
    T* ff_linear_weight_h;
    T* ff_linear_bias;

    FluxAttnPostProcessorWeight() = default;
    FluxAttnPostProcessorWeight(size_t embedding_dim, LyraQuantType quant_level, IAllocator* allocator);

    virtual ~FluxAttnPostProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
