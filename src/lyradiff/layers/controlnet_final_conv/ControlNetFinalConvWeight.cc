#include "ControlNetFinalConvWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ControlNetFinalConvWeight<T>::ControlNetFinalConvWeight(std::vector<size_t> block_out_channels)
{

    block_out_channels_ = block_out_channels;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ControlNetFinalConvWeight<T>::~ControlNetFinalConvWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // cout << "~ControlNetFinalConvWeight()" << endl;
    if (is_maintain_buffer) {
        for (int i = 0; i < conv_block_weights.size(); i++) {
            deviceFree(conv_block_weights[i]);
            deviceFree(conv_block_bias[i]);
        } 
    }
    // cout << "~ControlNetFinalConvWeight() finished" << endl;
}

template<typename T>
void ControlNetFinalConvWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        for (int i = 0; i < block_out_channels_.size(); i++) {
            // size_t channel_in = block_out_channels_[i];
            T* cur_conv_block_weight = nullptr;
            T* cur_conv_block_bias   = nullptr;
            deviceMalloc(&cur_conv_block_weight, block_out_channels_[i] * 1 * 1 * block_out_channels_[i]);
            deviceMalloc(&cur_conv_block_bias, block_out_channels_[i]);

            conv_block_weights.push_back(cur_conv_block_weight);
            conv_block_bias.push_back(cur_conv_block_bias);
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void ControlNetFinalConvWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    int    i          = 0;
    size_t channel_in = block_out_channels_[i];
    for (; i < block_out_channels_.size() - 1; i++) {
        channel_in = block_out_channels_[i];

        loadWeightFromBin<T>(conv_block_weights[i],
                             {channel_in * 1 * 1 * channel_in},
                             prefix + "controlnet_down_blocks." + std::to_string(i) + ".weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(conv_block_bias[i],
                             {channel_in},
                             prefix + "controlnet_down_blocks." + std::to_string(i) + ".bias.bin",
                             model_file_type);
    }
    channel_in = block_out_channels_[i];

    loadWeightFromBin<T>(conv_block_weights[i],
                         {channel_in * 1 * 1 * channel_in},
                         prefix + "controlnet_mid_block.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_block_bias[i], {channel_in}, prefix + "controlnet_mid_block.bias.bin", model_file_type);
    cout << "finished controlnet final conv weight load" << endl;
}

template<typename T>
void ControlNetFinalConvWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                        std::unordered_map<std::string, void*>& weights,
                                                        cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    int    i          = 0;
    size_t channel_in = block_out_channels_[i];
    for (; i < block_out_channels_.size() - 1; i++) {
        channel_in                   = block_out_channels_[i];
        void* tmp_conv_block_weights = weights[prefix + "controlnet_down_blocks." + std::to_string(i) + ".weight"];
        void* tmp_conv_block_bias    = weights[prefix + "controlnet_down_blocks." + std::to_string(i) + ".bias"];

        cudaMemcpy(
            conv_block_weights[i], tmp_conv_block_weights, sizeof(T) * channel_in * 1 * 1 * channel_in, memcpyKind);
        cudaMemcpy(conv_block_bias[i], tmp_conv_block_bias, sizeof(T) * channel_in, memcpyKind);
    }
    channel_in = block_out_channels_[i];

    void* tmp_conv_block_weights = weights[prefix + "controlnet_mid_block.weight"];
    void* tmp_conv_block_bias    = weights[prefix + "controlnet_mid_block.bias"];

    cudaMemcpy(conv_block_weights[i], tmp_conv_block_weights, sizeof(T) * channel_in * 1 * 1 * channel_in, memcpyKind);
    cudaMemcpy(conv_block_bias[i], tmp_conv_block_bias, sizeof(T) * channel_in, memcpyKind);
}

template class ControlNetFinalConvWeight<float>;
template class ControlNetFinalConvWeight<half>;
}  // namespace lyradiff