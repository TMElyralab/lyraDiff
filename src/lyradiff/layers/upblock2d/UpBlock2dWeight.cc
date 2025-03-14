#include "src/lyradiff/layers/upblock2d/UpBlock2dWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"
using namespace std;

namespace lyradiff {
template<typename T>
UpBlock2dWeight<T>::UpBlock2dWeight(const size_t in_channels,
                                    const size_t out_channels,
                                    const size_t prev_output_channel,
                                    const size_t norm_num_groups,
                                    IAllocator*  allocator):
    in_channels_(in_channels),
    out_channels_(out_channels),
    prev_output_channel_(prev_output_channel),
    norm_num_groups_(norm_num_groups)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_ = allocator;

    resnet_2d_block_weight1 =
        new Resnet2DBlockWeight<T>(out_channels_ + prev_output_channel_, out_channels_, true, allocator);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_ + out_channels_, out_channels_, true, allocator);
    resnet_2d_block_weight3 = new Resnet2DBlockWeight<T>(out_channels_ + in_channels_, out_channels_, true, allocator);

    upsampler_weight_size_ = out_channels_ * 3 * 3 * out_channels_;
    // TODO: 加入 resnet 2d block weights

    this->lora_layer_map = {{"resnets_0", resnet_2d_block_weight1},
                            {"resnets_1", resnet_2d_block_weight2},
                            {"resnets_2", resnet_2d_block_weight3}};
}

template<typename T>
UpBlock2dWeight<T>::~UpBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(upsampler_weight);
        deviceFree(upsampler_bias);

        upsampler_weight = nullptr;
        upsampler_bias   = nullptr;

        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;
        delete resnet_2d_block_weight3;

        resnet_2d_block_weight1 = nullptr;
        resnet_2d_block_weight2 = nullptr;
        resnet_2d_block_weight3 = nullptr;
    }
}

template<typename T>
void UpBlock2dWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    deviceMalloc(&upsampler_weight, out_channels_ * 3 * 3 * out_channels_);
    deviceMalloc(&upsampler_bias, out_channels_);

    // this->lora_weight_map      = {{"upsamplers_0_conv", upsampler_weight}};
    // this->lora_weight_size_map = {{"upsamplers_0_conv", out_channels_ * 3 * 3 * out_channels_}};

    this->lora_weight_map = {
        {"upsamplers_0_conv", new LoraWeight<T>({out_channels_, 3, 3, out_channels_}, upsampler_weight)}};

    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();
    resnet_2d_block_weight3->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void UpBlock2dWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    loadWeightFromBin<T>(upsampler_weight,
                         {out_channels_ * 3 * 3 * out_channels_},
                         prefix + "upsamplers.0.conv.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(upsampler_bias, {out_channels_}, prefix + "upsamplers.0.conv.bias.bin", model_file_type);

    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
    resnet_2d_block_weight3->loadWeights(prefix + std::string("resnets.2."), model_file_type);
}

template<typename T>
void UpBlock2dWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                              std::unordered_map<std::string, void*>& weights,
                                              cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_upsampler_weight = weights[prefix + "upsamplers.0.conv.weight"];
    void* tmp_upsampler_bias   = weights[prefix + "upsamplers.0.conv.bias"];

    weight_loader_manager_glob->doCudaMemcpy(upsampler_weight, tmp_upsampler_weight, sizeof(T) * out_channels_ * 3 * 3 * out_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(upsampler_bias, tmp_upsampler_bias, sizeof(T) * out_channels_, memcpyKind);

    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
    resnet_2d_block_weight3->loadWeightsFromCache(prefix + std::string("resnets.2."), weights, memcpyKind);
}

template<typename T>
void UpBlock2dWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&upsampler_weight_lora_buf_, upsampler_weight_size_);

        is_maintain_lora = true;
    }
}

template<typename T>
void UpBlock2dWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                            std::string                          prefix,
                                            std::unordered_map<std::string, T*>& lora_weights,
                                            float                                lora_alpha,
                                            FtCudaDataType                       lora_file_type,
                                            cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // for lcm-lora
    if (checkIfFileExist(lora_path + prefix + "upsamplers_0_conv.bin")) {
        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        loadWeightFromBin<T>(upsampler_weight_lora_buf_,
                             {upsampler_weight_size_},
                             lora_path + prefix + "upsamplers_0_conv.bin",
                             lora_file_type);

        T* upsamplers_0_conv = (T*)malloc(sizeof(T) * upsampler_weight_size_);
        cudaMemcpy(
            upsamplers_0_conv, upsampler_weight_lora_buf_, sizeof(T) * upsampler_weight_size_, cudaMemcpyDeviceToHost);
        lora_weights[prefix + "upsamplers_0_conv"] = upsamplers_0_conv;
        invokeLoadLora<T>(upsampler_weight, upsampler_weight_lora_buf_, upsampler_weight_size_, lora_alpha);
    }

    resnet_2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_2_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void UpBlock2dWeight<T>::loadLoraFromCache(std::string                          prefix,
                                           std::unordered_map<std::string, T*>& lora_weights,
                                           float                                lora_alpha,
                                           bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // for lcm-lora
    if (lora_weights.find(prefix + "upsamplers_0_conv") != lora_weights.end()) {

        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        T* upsamplers_0_conv = lora_weights[prefix + "upsamplers_0_conv"];
        cudaMemcpy(
            upsampler_weight_lora_buf_, upsamplers_0_conv, sizeof(T) * upsampler_weight_size_, cudaMemcpyHostToDevice);
        invokeLoadLora<T>(upsampler_weight, upsampler_weight_lora_buf_, upsampler_weight_size_, lora_alpha);
    }

    resnet_2d_block_weight1->loadLoraFromCache(
        prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight2->loadLoraFromCache(
        prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight3->loadLoraFromCache(
        prefix + std::string("resnets_2_"), lora_weights, lora_alpha, from_outside);
}

template class UpBlock2dWeight<float>;
template class UpBlock2dWeight<half>;
}  // namespace lyradiff