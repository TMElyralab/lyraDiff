#pragma once

#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class UpDecoderBlock2dWeight {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t norm_num_groups_;
    size_t temb_channels_;
    bool   add_upsample_;

protected:
    bool is_maintain_buffer = false;

public:
    T* upsampler_weight = nullptr;
    T* upsampler_bias   = nullptr;

    Resnet2DBlockWeight<T>* resnet_2d_block_weight1 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight2 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight3 = nullptr;

    UpDecoderBlock2dWeight() = default;
    UpDecoderBlock2dWeight(const size_t in_channels,
                           const size_t out_channels,
                           const size_t norm_num_groups,
                           const size_t temb_channels,
                           const bool   add_upsample);

    ~UpDecoderBlock2dWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void mallocWeights();
    virtual void
    loadWeightsFromCache(std::string prefix, std::unordered_map<std::string, void*>& weights, cudaMemcpyKind memcpyKind);
};

}  // namespace lyradiff
