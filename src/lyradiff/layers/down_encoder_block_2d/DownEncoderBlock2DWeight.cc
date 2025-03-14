#include "DownEncoderBlock2DWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {
template<typename T>
DownEncoderBlock2DWeight<T>::DownEncoderBlock2DWeight(const size_t in_channels,
                                                      const size_t out_channels,
                                                      const size_t norm_num_groups,
                                                      const size_t temb_channels,
                                                      const bool   add_downsample):
    in_channels_(in_channels),
    out_channels_(out_channels),
    norm_num_groups_(norm_num_groups),
    temb_channels_(temb_channels),
    add_downsample_(add_downsample)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    resnet_2d_block_weight1 = new Resnet2DBlockWeight<T>(in_channels, out_channels_, temb_channels_ > 0);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_, out_channels_, temb_channels_ > 0);

    // TODO: 加入 resnet 2d block weights
}

template<typename T>
DownEncoderBlock2DWeight<T>::~DownEncoderBlock2DWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        if (add_downsample_) {
            deviceFree(downsampler_weight);
            deviceFree(downsampler_bias);
        }

        downsampler_weight = nullptr;
        downsampler_bias   = nullptr;

        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;

        resnet_2d_block_weight1 = nullptr;
        resnet_2d_block_weight2 = nullptr;
    }
}

template<typename T>
void DownEncoderBlock2DWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (add_downsample_) {
        deviceMalloc(&downsampler_weight, out_channels_ * 3 * 3 * out_channels_);
        deviceMalloc(&downsampler_bias, out_channels_);
    }

    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void DownEncoderBlock2DWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    if (add_downsample_) {
        loadWeightFromBin<T>(downsampler_weight,
                             {out_channels_ * 3 * 3 * out_channels_},
                             prefix + "downsamplers.0.conv.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(
            downsampler_bias, {out_channels_}, prefix + "downsamplers.0.conv.bias.bin", model_file_type);
    }

    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
}

template<typename T>
void DownEncoderBlock2DWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                       std::unordered_map<std::string, void*>& weights,
                                                       cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    if (add_downsample_) {
        void* tmp_downsampler_weight = weights[prefix + "downsamplers.0.conv.weight"];
        void* tmp_downsampler_bias   = weights[prefix + "downsamplers.0.conv.bias"];

        weight_loader_manager_glob->doCudaMemcpy(
            downsampler_weight, tmp_downsampler_weight, sizeof(T) * out_channels_ * 3 * 3 * out_channels_, memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(downsampler_bias, tmp_downsampler_bias, sizeof(T) * out_channels_, memcpyKind);
    }
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
}

template class DownEncoderBlock2DWeight<float>;
template class DownEncoderBlock2DWeight<half>;
}  // namespace lyradiff