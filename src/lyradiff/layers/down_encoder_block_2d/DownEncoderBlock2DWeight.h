#pragma once

#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class DownEncoderBlock2DWeight {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t norm_num_groups_;
    size_t temb_channels_;
    bool   add_downsample_;

protected:
    bool is_maintain_buffer = false;

public:
    T* downsampler_weight = nullptr;
    T* downsampler_bias   = nullptr;

    Resnet2DBlockWeight<T>* resnet_2d_block_weight1 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight2 = nullptr;

    DownEncoderBlock2DWeight() = default;
    DownEncoderBlock2DWeight(const size_t in_channels,
                             const size_t out_channels,
                             const size_t norm_num_groups,
                             const size_t temb_channels,
                             const bool   add_downsample);

    ~DownEncoderBlock2DWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void mallocWeights();
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
};

}  // namespace lyradiff
