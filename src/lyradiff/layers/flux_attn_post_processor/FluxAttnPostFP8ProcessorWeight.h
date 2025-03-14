#pragma once
#include "FluxAttnPostProcessorWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxAttnPostFP8ProcessorWeight: public FluxAttnPostProcessorWeight<T> {
private:
    size_t embedding_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    __nv_fp8_e4m3* gelu_proj_weight;
    T*             gelu_proj_weight_h;
    T*             gelu_proj_bias;
    float*         gelu_proj_weight_scale;
    float*         gelu_proj_input_scale;

    __nv_fp8_e4m3* ff_linear_weight;
    T*             ff_linear_weight_h;
    T*             ff_linear_bias;
    float*         ff_linear_weight_scale;
    float*         ff_linear_input_scale;

    FluxAttnPostFP8ProcessorWeight() = default;
    FluxAttnPostFP8ProcessorWeight(size_t embedding_dim, LyraQuantType quant_level, IAllocator* allocator);

    virtual ~FluxAttnPostFP8ProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
