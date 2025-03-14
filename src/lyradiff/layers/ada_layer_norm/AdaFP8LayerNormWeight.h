#pragma once
#include "AdaLayerNormWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
namespace lyradiff {

template<typename T>
class AdaFP8LayerNormWeight: public AdaLayerNormWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_scale_;

protected:
    bool is_maintain_buffer = false;

public:
    __nv_fp8_e4m3* linear_weight;
    T*             linear_bias;
    T*             linear_weight_h;  // on host original weight

    float* linear_weight_scale;
    float* linear_input_scale;

    AdaFP8LayerNormWeight() = default;
    AdaFP8LayerNormWeight(size_t        embedding_dim,
                          size_t        embedding_scale,
                          LyraQuantType quant_level,
                          IAllocator*   allocator = nullptr);

    virtual ~AdaFP8LayerNormWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
