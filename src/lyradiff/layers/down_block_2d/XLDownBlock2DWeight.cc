#include "XLDownBlock2DWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLDownBlock2DWeight<T>::XLDownBlock2DWeight(const size_t in_channels_,
                                            const size_t inter_channels_,
                                            const size_t out_channels_,
                                            const bool   is_downsampler_,
                                            IAllocator*  allocator):
    in_channels_(in_channels_),
    inter_channels_(inter_channels_),
    out_channels_(out_channels_),
    is_downsampler_(is_downsampler_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_ = allocator;
    resnet_0_weights = new Resnet2DBlockWeight<T>(in_channels_, inter_channels_, true, allocator);
    resnet_1_weights = new Resnet2DBlockWeight<T>(inter_channels_, out_channels_, true, allocator);

    this->lora_layer_map = {{"resnets_0", resnet_0_weights}, {"resnets_1", resnet_1_weights}};
}

template<typename T>
XLDownBlock2DWeight<T>::~XLDownBlock2DWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        if (is_downsampler_) {
            deviceFree(downsampler_weight);
            deviceFree(downsampler_bias);

            downsampler_weight = nullptr;
            downsampler_bias   = nullptr;
        }

        delete resnet_0_weights;
        delete resnet_1_weights;

        resnet_0_weights = nullptr;
        resnet_1_weights = nullptr;
    }

    if (is_maintain_lora && is_downsampler_) {
        deviceFree(downsampler_weight_lora_buf);
    }
}

template<typename T>
void XLDownBlock2DWeight<T>::mallocWeights()
{
    if (is_downsampler_) {
        deviceMalloc(&downsampler_weight, out_channels_ * 3 * 3 * out_channels_);
        deviceMalloc(&downsampler_bias, out_channels_);
    }
    this->lora_weight_map = {
        {"downsamplers_0_conv", new LoraWeight<T>({out_channels_, 3, 3, out_channels_}, downsampler_weight)}};

    resnet_0_weights->mallocWeights();
    resnet_1_weights->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void XLDownBlock2DWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora && is_downsampler_) {
        deviceMalloc(&downsampler_weight_lora_buf, out_channels_ * 3 * 3 * out_channels_);

        is_maintain_lora = true;
    }
}

template<typename T>
void XLDownBlock2DWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    if (is_downsampler_) {
        loadWeightFromBin<T>(downsampler_weight,
                             {out_channels_ * 3 * 3 * out_channels_},
                             prefix + "downsamplers.0.conv.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(
            downsampler_bias, {out_channels_}, prefix + "downsamplers.0.conv.bias.bin", model_file_type);
    }
    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights->loadWeights(resnet_0_model_prefix_, model_file_type);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights->loadWeights(resnet_1_model_prefix_, model_file_type);
}

template<typename T>
void XLDownBlock2DWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                  std::unordered_map<std::string, void*>& weights,
                                                  cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    if (is_downsampler_) {
        void* tmp_downsampler_weight = weights[prefix + "downsamplers.0.conv.weight"];
        void* tmp_downsampler_bias   = weights[prefix + "downsamplers.0.conv.bias"];

        cudaMemcpy(
            downsampler_weight, tmp_downsampler_weight, sizeof(T) * out_channels_ * 3 * 3 * out_channels_, memcpyKind);
        cudaMemcpy(downsampler_bias, tmp_downsampler_bias, sizeof(T) * out_channels_, memcpyKind);
    }

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights->loadWeightsFromCache(resnet_0_model_prefix_, weights, memcpyKind);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights->loadWeightsFromCache(resnet_1_model_prefix_, weights, memcpyKind);
}

template<typename T>
void XLDownBlock2DWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                std::string                          prefix,
                                                std::unordered_map<std::string, T*>& lora_weights,
                                                float                                lora_alpha,
                                                FtCudaDataType                       lora_file_type,
                                                cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }
    if (is_downsampler_) {
        loadWeightFromBin<T>(downsampler_weight_lora_buf,
                             {out_channels_ * 3 * 3 * out_channels_},
                             lora_path + prefix + "downsamplers_0_conv.bin",
                             lora_file_type);

        T* conv = (T*)malloc(sizeof(T) * out_channels_ * 3 * 3 * out_channels_);

        lora_weights[prefix + "downsamplers_0_conv"] = conv;

        // 再缓存到本地
        cudaMemcpy(conv,
                   downsampler_weight_lora_buf,
                   sizeof(T) * out_channels_ * 3 * 3 * out_channels_,
                   cudaMemcpyDeviceToHost);

        invokeLoadLora<T>(
            downsampler_weight, downsampler_weight_lora_buf, out_channels_ * 3 * 3 * out_channels_, lora_alpha);
    }
    resnet_0_weights->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_1_weights->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void XLDownBlock2DWeight<T>::loadLoraFromCache(std::string                          prefix,
                                               std::unordered_map<std::string, T*>& lora_weights,
                                               float                                lora_alpha,
                                               bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }
    if (is_downsampler_) {
        T* conv = lora_weights[prefix + "downsamplers_0_conv"];
        cudaMemcpy(downsampler_weight_lora_buf,
                   conv,
                   sizeof(T) * out_channels_ * 3 * 3 * out_channels_,
                   cudaMemcpyHostToDevice);
        invokeLoadLora<T>(
            downsampler_weight, downsampler_weight_lora_buf, out_channels_ * 3 * 3 * out_channels_, lora_alpha);
    }
    resnet_0_weights->loadLoraFromCache(prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_1_weights->loadLoraFromCache(prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
}

template class XLDownBlock2DWeight<float>;
template class XLDownBlock2DWeight<half>;
}  // namespace lyradiff