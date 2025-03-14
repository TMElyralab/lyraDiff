#include "UpDecoderBlock2dWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {
template<typename T>
UpDecoderBlock2dWeight<T>::UpDecoderBlock2dWeight(const size_t in_channels,
                                                  const size_t out_channels,
                                                  const size_t norm_num_groups,
                                                  const size_t temb_channels,
                                                  const bool   add_upsample):
    in_channels_(in_channels),
    out_channels_(out_channels),
    norm_num_groups_(norm_num_groups),
    temb_channels_(temb_channels),
    add_upsample_(add_upsample)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    resnet_2d_block_weight1 = new Resnet2DBlockWeight<T>(in_channels, out_channels_, temb_channels_ > 0);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_, out_channels_, temb_channels_ > 0);
    resnet_2d_block_weight3 = new Resnet2DBlockWeight<T>(out_channels_, out_channels_, temb_channels_ > 0);

    // TODO: 加入 resnet 2d block weights
}

template<typename T>
UpDecoderBlock2dWeight<T>::~UpDecoderBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        if (add_upsample_) {
            deviceFree(upsampler_weight);
            deviceFree(upsampler_bias);
        }

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
void UpDecoderBlock2dWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (add_upsample_) {
        deviceMalloc(&upsampler_weight, out_channels_ * 3 * 3 * out_channels_);
        deviceMalloc(&upsampler_bias, out_channels_);
    }

    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();
    resnet_2d_block_weight3->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void UpDecoderBlock2dWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (add_upsample_) {
        loadWeightFromBin<T>(upsampler_weight,
                             {out_channels_ * 3 * 3 * out_channels_},
                             prefix + "upsamplers.0.conv.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(upsampler_bias, {out_channels_}, prefix + "upsamplers.0.conv.bias.bin", model_file_type);
    }

    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
    resnet_2d_block_weight3->loadWeights(prefix + std::string("resnets.2."), model_file_type);
}

template<typename T>
void UpDecoderBlock2dWeight<T>::loadWeightsFromCache(std::string                          prefix,
                                                     std::unordered_map<std::string, void*>& weights,
                                                     cudaMemcpyKind                       memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    if (add_upsample_) {
        void* tmp_upsampler_weight = weights[prefix + "upsamplers.0.conv.weight"];
        void* tmp_upsampler_bias   = weights[prefix + "upsamplers.0.conv.bias"];

        weight_loader_manager_glob->doCudaMemcpy(upsampler_weight, tmp_upsampler_weight, sizeof(T) * out_channels_ * 3 * 3 * out_channels_, memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(upsampler_bias, tmp_upsampler_bias, sizeof(T) * out_channels_, memcpyKind);
    }
    
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
    resnet_2d_block_weight3->loadWeightsFromCache(prefix + std::string("resnets.2."), weights, memcpyKind);
}

template class UpDecoderBlock2dWeight<float>;
template class UpDecoderBlock2dWeight<half>;
}  // namespace lyradiff