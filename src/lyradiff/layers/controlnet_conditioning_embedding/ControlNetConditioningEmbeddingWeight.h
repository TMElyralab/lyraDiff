#pragma once
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class ControlNetConditioningEmbeddingWeight {
private:
    std::vector<size_t> block_out_channels_ = {16, 32, 96, 256};
    size_t              conditioning_channels_;
    size_t              conditioning_embedding_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    std::vector<T*> conv_block_weights = std::vector<T*>(6, nullptr);
    std::vector<T*> conv_block_bias    = std::vector<T*>(6, nullptr);

    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    T* conv_out_weight = nullptr;
    T* conv_out_bias   = nullptr;

    ControlNetConditioningEmbeddingWeight() = default;
    ControlNetConditioningEmbeddingWeight(const size_t conditioning_channels,
                                          const size_t conditioning_embedding_channels);

    ~ControlNetConditioningEmbeddingWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
