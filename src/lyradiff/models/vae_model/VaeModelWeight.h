#pragma once
#include "src/lyradiff/layers/vae_decoder/VaeDecoderWeight.h"
#include "src/lyradiff/layers/vae_encoder/VaeEncoderWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>
#include <vector>

namespace lyradiff {
template<typename T>
class VaeModelWeight {
private:
    size_t norm_num_groups_ = 32;
    size_t input_channels_  = 3;
    size_t output_channels_ = 3;
    size_t latent_channels_ = 4;

protected:
    bool is_maintain_encoder_buffer = false;
    bool is_maintain_decoder_buffer = false;
    bool is_malloced                = false;
    bool is_upcast_                 = false;

public:
    T* post_quant_conv_weight = nullptr;
    T* post_quant_conv_bias   = nullptr;

    T* quant_conv_weight = nullptr;
    T* quant_conv_bias   = nullptr;

    VaeDecoderWeight<T> vae_decoder_weight;
    VaeEncoderWeight<T> vae_encoder_weight;

    // VaeModelWeight() = default;

    VaeModelWeight(const bool is_upcast = false);

    ~VaeModelWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void loadEncoderWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadEncoderWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind);

    virtual void loadDecoderWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadDecoderWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
    virtual void mallocEncoderWeights();
    virtual void mallocDecoderWeights();

    virtual void loadS3DiffLoraFromStateDict(std::unordered_map<std::string, T*>& lora_weights, bool is_alpha);
};

}  // namespace lyradiff
