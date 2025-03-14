#include "DownBlock2DWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
DownBlock2DWeight<T>::DownBlock2DWeight(const size_t in_channels_,
                                        const size_t inter_channels_,
                                        const size_t out_channels_,
                                        IAllocator*  allocator):
    in_channels_(in_channels_), inter_channels_(inter_channels_), out_channels_(out_channels_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_ = allocator;
    resnet_0_weights = new Resnet2DBlockWeight<T>(in_channels_, inter_channels_, true, allocator);
    resnet_1_weights = new Resnet2DBlockWeight<T>(inter_channels_, out_channels_, true, allocator);

    this->lora_layer_map = {{"resnets_0", resnet_0_weights}, {"resnets_1", resnet_1_weights}};
}

template<typename T>
DownBlock2DWeight<T>::~DownBlock2DWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete resnet_0_weights;
    delete resnet_1_weights;
    resnet_0_weights = nullptr;
    resnet_1_weights = nullptr;
}

template<typename T>
void DownBlock2DWeight<T>::mallocWeights()
{
    resnet_0_weights->mallocWeights();
    resnet_1_weights->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void DownBlock2DWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights->loadWeights(resnet_0_model_prefix_, model_file_type);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights->loadWeights(resnet_1_model_prefix_, model_file_type);
}

template<typename T>
void DownBlock2DWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                std::unordered_map<std::string, void*>& weights,
                                                cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights->loadWeightsFromCache(resnet_0_model_prefix_, weights, memcpyKind);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights->loadWeightsFromCache(resnet_1_model_prefix_, weights, memcpyKind);
}

template<typename T>
void DownBlock2DWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                              std::string                          prefix,
                                              std::unordered_map<std::string, T*>& lora_weights,
                                              float                                lora_alpha,
                                              FtCudaDataType                       lora_file_type,
                                              cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    resnet_0_weights->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_1_weights->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void DownBlock2DWeight<T>::loadLoraFromCache(std::string                          prefix,
                                             std::unordered_map<std::string, T*>& lora_weights,
                                             float                                lora_alpha,
                                             bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    resnet_0_weights->loadLoraFromCache(prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_1_weights->loadLoraFromCache(prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
}

template class DownBlock2DWeight<float>;
template class DownBlock2DWeight<half>;
}  // namespace lyradiff