#pragma once
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class ZeroSFTWeight {
private:
    size_t cond_channels_;
    size_t project_channels_;
    size_t concat_channels_;
    size_t nhidden = 128;

protected:
    bool is_maintain_buffer = false;

public:
    T* mlp_conv_weight;
    T* mlp_conv_bias;
    T* zero_conv_weight;
    T* zero_conv_bias;
    T* zero_mul_weight;
    T* zero_mul_bias;
    T* zero_add_weight;
    T* zero_add_bias;
    T* norm_gamma;
    T* norm_beta;

    ZeroSFTWeight() = default;
    ZeroSFTWeight(size_t project_channels, size_t cond_channels, size_t concat_channels);

    ~ZeroSFTWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();

};

}  // namespace lyradiff
