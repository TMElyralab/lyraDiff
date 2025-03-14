#include "VaeModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {

template<typename T>
VaeModelWeight<T>::VaeModelWeight(const bool is_upcast)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    is_upcast_ = is_upcast;

    vae_decoder_weight = VaeDecoderWeight<T>(latent_channels_, output_channels_, norm_num_groups_, is_upcast);
    vae_encoder_weight = VaeEncoderWeight<T>(input_channels_, latent_channels_, norm_num_groups_);
}

template<typename T>
VaeModelWeight<T>::~VaeModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_encoder_buffer) {
        deviceFree(quant_conv_weight);
        deviceFree(quant_conv_bias);

        quant_conv_weight = nullptr;
        quant_conv_bias   = nullptr;
    }

    if (is_maintain_decoder_buffer) {
        deviceFree(post_quant_conv_weight);
        deviceFree(post_quant_conv_bias);

        post_quant_conv_weight = nullptr;
        post_quant_conv_bias   = nullptr;
    }
}

template<typename T>
void VaeModelWeight<T>::mallocWeights()
{
    mallocEncoderWeights();
    mallocDecoderWeights();
}

template<typename T>
void VaeModelWeight<T>::mallocEncoderWeights()
{
    if (!is_maintain_encoder_buffer) {
        deviceMalloc(&quant_conv_weight, latent_channels_ * 2 * 1 * 1 * latent_channels_ * 2);
        deviceMalloc(&quant_conv_bias, latent_channels_ * 2);
        vae_encoder_weight.mallocWeights();
    }

    is_maintain_encoder_buffer = true;
}

template<typename T>
void VaeModelWeight<T>::mallocDecoderWeights()
{
    if (!is_maintain_decoder_buffer) {
        deviceMalloc(&post_quant_conv_weight, latent_channels_ * 1 * 1 * latent_channels_);
        deviceMalloc(&post_quant_conv_bias, latent_channels_);

        vae_decoder_weight.mallocWeights();
    }

    is_maintain_decoder_buffer = true;
}

template<typename T>
void VaeModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    loadEncoderWeights(prefix, model_file_type);
    loadDecoderWeights(prefix, model_file_type);
}

template<typename T>
void VaeModelWeight<T>::loadEncoderWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_encoder_buffer) {
        mallocEncoderWeights();
    }

    loadWeightFromBin<T>(quant_conv_weight,
                         {latent_channels_ * 2 * 1 * 1 * latent_channels_ * 2},
                         prefix + "quant_conv.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(quant_conv_bias, {latent_channels_ * 2}, prefix + "quant_conv.bias.bin", model_file_type);

    vae_encoder_weight.loadWeights(prefix + "encoder.", model_file_type);
}

template<typename T>
void VaeModelWeight<T>::loadDecoderWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_decoder_buffer) {
        mallocDecoderWeights();
    }

    loadWeightFromBin<T>(post_quant_conv_weight,
                         {latent_channels_ * 1 * 1 * latent_channels_},
                         prefix + "post_quant_conv.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        post_quant_conv_bias, {latent_channels_}, prefix + "post_quant_conv.bias.bin", model_file_type);

    vae_decoder_weight.loadWeights(prefix + "decoder.", model_file_type);
}

template<typename T>
void VaeModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    loadEncoderWeightsFromCache(prefix, weights, memcpyKind);
    loadDecoderWeightsFromCache(prefix, weights, memcpyKind);
}

template<typename T>
void VaeModelWeight<T>::loadEncoderWeightsFromCache(std::string                             prefix,
                                                    std::unordered_map<std::string, void*>& weights,
                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_encoder_buffer) {
        mallocEncoderWeights();
    }

    void* tmp_quant_conv_weight = weights[prefix + "quant_conv.weight"];
    void* tmp_quant_conv_bias   = weights[prefix + "quant_conv.bias"];

    weight_loader_manager_glob->doCudaMemcpy(quant_conv_weight,
                                             tmp_quant_conv_weight,
                                             sizeof(T) * latent_channels_ * 2 * 1 * 1 * latent_channels_ * 2,
                                             memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        quant_conv_bias, tmp_quant_conv_bias, sizeof(T) * latent_channels_ * 2, memcpyKind);

    vae_encoder_weight.loadWeightsFromCache(prefix + "encoder.", weights, memcpyKind);
}

template<typename T>
void VaeModelWeight<T>::loadDecoderWeightsFromCache(std::string                             prefix,
                                                    std::unordered_map<std::string, void*>& weights,
                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_decoder_buffer) {
        mallocDecoderWeights();
    }

    void* tmp_post_quant_conv_weight = weights[prefix + "post_quant_conv.weight"];
    void* tmp_post_quant_conv_bias   = weights[prefix + "post_quant_conv.bias"];

    weight_loader_manager_glob->doCudaMemcpy(post_quant_conv_weight,
                                             tmp_post_quant_conv_weight,
                                             sizeof(T) * latent_channels_ * 1 * 1 * latent_channels_,
                                             memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        post_quant_conv_bias, tmp_post_quant_conv_bias, sizeof(T) * latent_channels_, memcpyKind);
    vae_decoder_weight.loadWeightsFromCache(prefix + "decoder.", weights, memcpyKind);
}

template<typename T>
void VaeModelWeight<T>::loadS3DiffLoraFromStateDict(std::unordered_map<std::string, T*>& lora_weights, bool is_alpha)
{
    auto& lora_container = weight_loader_manager_glob->map_lora_container["vae"];
    for (auto iter : lora_weights) {
        lora_container.add_lora_weight(iter.first, iter.second, is_alpha);
    }
}

template class VaeModelWeight<float>;
template class VaeModelWeight<half>;
}  // namespace lyradiff