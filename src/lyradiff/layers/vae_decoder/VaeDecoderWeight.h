#pragma once

#include "src/lyradiff/layers/unet_mid_block_2d/UNetMidBlock2DWeight.h"
#include "src/lyradiff/layers/up_decoder_block_2d/UpDecoderBlock2dWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class VaeDecoderWeight {
private:
    std::vector<size_t> block_out_channels_ = {512, 512, 256, 128};
    size_t              temb_channels_      = 0;
    size_t              norm_num_groups_;
    size_t              in_channels_;
    size_t              out_channels_;
    bool                is_upcast_;

protected:
    bool is_maintain_buffer = false;

public:
    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    T* conv_norm_out_gamma = nullptr;
    T* conv_norm_out_beta  = nullptr;

    T* conv_out_weight = nullptr;
    T* conv_out_bias   = nullptr;

    float* upcast_conv_norm_out_gamma = nullptr;
    float* upcast_conv_norm_out_beta  = nullptr;

    UNetMidBlock2DWeight<T>* unet_mid_block_2d_weight = nullptr;

    UpDecoderBlock2dWeight<T>* up_decoder_block_2d_weight_0 = nullptr;
    UpDecoderBlock2dWeight<T>* up_decoder_block_2d_weight_1 = nullptr;
    UpDecoderBlock2dWeight<T>* up_decoder_block_2d_weight_2 = nullptr;
    UpDecoderBlock2dWeight<T>* up_decoder_block_2d_weight_3 = nullptr;

    UpDecoderBlock2dWeight<float>* upcast_up_decoder_block_2d_weight_2 = nullptr;
    UpDecoderBlock2dWeight<float>* upcast_up_decoder_block_2d_weight_3 = nullptr;

    VaeDecoderWeight() = default;
    VaeDecoderWeight(const size_t in_channels,
                     const size_t out_channels,
                     const size_t norm_num_groups,
                     const bool   is_upcast = false);

    ~VaeDecoderWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void
    loadWeightsFromCache(std::string prefix, std::unordered_map<std::string, void*>& weights, cudaMemcpyKind memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
