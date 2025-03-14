#include "ControlNetConditioningEmbeddingWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ControlNetConditioningEmbeddingWeight<T>::ControlNetConditioningEmbeddingWeight(
    const size_t conditioning_channels, const size_t conditioning_embedding_channels):
    conditioning_channels_(conditioning_channels), conditioning_embedding_channels_(conditioning_embedding_channels)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ControlNetConditioningEmbeddingWeight<T>::~ControlNetConditioningEmbeddingWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);
        deviceFree(conv_out_weight);
        deviceFree(conv_out_bias);

        conv_in_weight = nullptr;
        conv_in_bias   = nullptr;

        conv_out_weight = nullptr;
        conv_out_bias   = nullptr;

        for (int i = 0; i < conv_block_weights.size(); i++) {
            deviceFree(conv_block_weights[i]);
            deviceFree(conv_block_bias[i]);

            conv_block_weights[i] = nullptr;
            conv_block_bias[i]    = nullptr;
        }
    }
}

template<typename T>
void ControlNetConditioningEmbeddingWeight<T>::mallocWeights()
{
    int i = 0;
    deviceMalloc(&conv_in_weight, conditioning_channels_ * 3 * 3 * block_out_channels_[i]);
    deviceMalloc(&conv_in_bias, block_out_channels_[i]);

    for (; i < block_out_channels_.size() - 1; i++) {
        size_t channel_in  = block_out_channels_[i];
        size_t channel_out = block_out_channels_[i + 1];

        deviceMalloc(&conv_block_weights[i * 2], channel_in * 3 * 3 * channel_in);
        deviceMalloc(&conv_block_bias[i * 2], channel_in);

        deviceMalloc(&conv_block_weights[i * 2 + 1], channel_in * 3 * 3 * channel_out);
        deviceMalloc(&conv_block_bias[i * 2 + 1], channel_out);
    }

    deviceMalloc(&conv_out_weight, block_out_channels_[i] * 3 * 3 * conditioning_embedding_channels_);
    deviceMalloc(&conv_out_bias, conditioning_embedding_channels_);

    is_maintain_buffer = true;
}

template<typename T>
void ControlNetConditioningEmbeddingWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    int i = 0;

    loadWeightFromBin<T>(conv_in_weight,
                         {block_out_channels_[i] * 3 * 3 * conditioning_channels_},
                         prefix + "conv_in.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_in_bias, {block_out_channels_[i]}, prefix + "conv_in.bias.bin", model_file_type);

    for (; i < block_out_channels_.size() - 1; i++) {
        size_t channel_in  = block_out_channels_[i];
        size_t channel_out = block_out_channels_[i + 1];

        loadWeightFromBin<T>(conv_block_weights[i * 2],
                             {channel_in * 3 * 3 * channel_in},
                             prefix + "blocks." + std::to_string(i * 2) + ".weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(conv_block_bias[i * 2],
                             {channel_in},
                             prefix + "blocks." + std::to_string(i * 2) + ".bias.bin",
                             model_file_type);

        loadWeightFromBin<T>(conv_block_weights[i * 2 + 1],
                             {channel_out * 3 * 3 * channel_in},
                             prefix + "blocks." + std::to_string(i * 2 + 1) + ".weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(conv_block_bias[i * 2 + 1],
                             {channel_out},
                             prefix + "blocks." + std::to_string(i * 2 + 1) + ".bias.bin",
                             model_file_type);
    }

    loadWeightFromBin<T>(conv_out_weight,
                         {conditioning_embedding_channels_ * 3 * 3 * block_out_channels_[i]},
                         prefix + "conv_out.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        conv_out_bias, {conditioning_embedding_channels_}, prefix + "conv_out.bias.bin", model_file_type);
}

template<typename T>
void ControlNetConditioningEmbeddingWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                                    std::unordered_map<std::string, void*>& weights,
                                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    int i = 0;

    void* tmp_conv_block_weights = weights[prefix + "conv_in.weight"];
    void* tmp_conv_block_bias    = weights[prefix + "conv_in.bias"];

    cudaMemcpy(conv_in_weight,
               tmp_conv_block_weights,
               sizeof(T) * block_out_channels_[i] * 3 * 3 * conditioning_channels_,
               memcpyKind);
    cudaMemcpy(conv_in_bias, tmp_conv_block_bias, sizeof(T) * block_out_channels_[i], memcpyKind);

    for (; i < block_out_channels_.size() - 1; i++) {
        size_t channel_in  = block_out_channels_[i];
        size_t channel_out = block_out_channels_[i + 1];

        void* tmp_conv_block_weights_1 = weights[prefix + "blocks." + std::to_string(i * 2) + ".weight"];
        void* tmp_conv_block_bias_1    = weights[prefix + "blocks." + std::to_string(i * 2) + ".bias"];

        cudaMemcpy(conv_block_weights[i * 2],
                   tmp_conv_block_weights_1,
                   sizeof(T) * channel_in * 3 * 3 * channel_in,
                   memcpyKind);
        cudaMemcpy(conv_block_bias[i * 2], tmp_conv_block_bias_1, sizeof(T) * channel_in, memcpyKind);

        void* tmp_conv_block_weights_2 = weights[prefix + "blocks." + std::to_string(i * 2 + 1) + ".weight"];
        void* tmp_conv_block_bias_2    = weights[prefix + "blocks." + std::to_string(i * 2 + 1) + ".bias"];

        cudaMemcpy(conv_block_weights[i * 2 + 1],
                   tmp_conv_block_weights_2,
                   sizeof(T) * channel_out * 3 * 3 * channel_in,
                   memcpyKind);
        cudaMemcpy(conv_block_bias[i * 2 + 1], tmp_conv_block_bias_2, sizeof(T) * channel_out, memcpyKind);
    }

    void* tmp_conv_block_weights_final = weights[prefix + "conv_out.weight"];
    void* tmp_conv_block_bias_final    = weights[prefix + "conv_out.bias"];

    cudaMemcpy(conv_out_weight,
               tmp_conv_block_weights_final,
               sizeof(T) * conditioning_embedding_channels_ * 3 * 3 * block_out_channels_[i],
               memcpyKind);
    cudaMemcpy(conv_out_bias, tmp_conv_block_bias_final, sizeof(T) * conditioning_embedding_channels_, memcpyKind);
}

template class ControlNetConditioningEmbeddingWeight<float>;
template class ControlNetConditioningEmbeddingWeight<half>;
}  // namespace lyradiff