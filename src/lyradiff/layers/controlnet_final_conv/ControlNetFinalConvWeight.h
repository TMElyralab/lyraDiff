#pragma once
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class ControlNetFinalConvWeight {
private:
    std::vector<size_t> block_out_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    std::vector<T*> conv_block_weights;
    std::vector<T*> conv_block_bias;

    ControlNetFinalConvWeight() = default;
    ControlNetFinalConvWeight(std::vector<size_t> block_out_channels);

    ~ControlNetFinalConvWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
