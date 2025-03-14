#pragma once

#include "src/lyradiff/layers/down_encoder_block_2d/DownEncoderBlock2DWeight.h"
#include "src/lyradiff/layers/unet_mid_block_2d/UNetMidBlock2DWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class VaeEncoderWeight {
private:
    std::vector<size_t> block_out_channels_ = {128, 128, 256, 512};
    size_t              temb_channels_      = 0;
    size_t              norm_num_groups_;
    size_t              in_channels_;
    size_t              out_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    T* conv_norm_out_gamma = nullptr;
    T* conv_norm_out_beta  = nullptr;

    T* conv_out_weight = nullptr;
    T* conv_out_bias   = nullptr;

    UNetMidBlock2DWeight<T>* unet_mid_block_2d_weight = nullptr;

    DownEncoderBlock2DWeight<T>* down_encoder_block_2d_weight_0 = nullptr;
    DownEncoderBlock2DWeight<T>* down_encoder_block_2d_weight_1 = nullptr;
    DownEncoderBlock2DWeight<T>* down_encoder_block_2d_weight_2 = nullptr;
    DownEncoderBlock2DWeight<T>* down_encoder_block_2d_weight_3 = nullptr;

    VaeEncoderWeight() = default;
    VaeEncoderWeight(const size_t in_channels, const size_t out_channels, const size_t norm_num_groups);

    ~VaeEncoderWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
