#include "src/lyradiff/layers/upblock2d/XLUpBlock2dWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {
template<typename T>
XLUpBlock2dWeight<T>::XLUpBlock2dWeight(const size_t in_channels,
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

    this->lora_layer_map = {{"resnets_0", resnet_2d_block_weight1},
                            {"resnets_1", resnet_2d_block_weight2},
                            {"resnets_2", resnet_2d_block_weight3}};
}

template<typename T>
XLUpBlock2dWeight<T>::~XLUpBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;
        delete resnet_2d_block_weight3;

        resnet_2d_block_weight1 = nullptr;
        resnet_2d_block_weight2 = nullptr;
        resnet_2d_block_weight3 = nullptr;
    }
}

template<typename T>
void XLUpBlock2dWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();
    resnet_2d_block_weight3->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void XLUpBlock2dWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
    resnet_2d_block_weight3->loadWeights(prefix + std::string("resnets.2."), model_file_type);
}

template<typename T>
void XLUpBlock2dWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                std::unordered_map<std::string, void*>& weights,
                                                cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
    resnet_2d_block_weight3->loadWeightsFromCache(prefix + std::string("resnets.2."), weights, memcpyKind);
}

template<typename T>
void XLUpBlock2dWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        // deviceMalloc(&upsampler_weight_lora_buf_, upsampler_weight_size_);

        is_maintain_lora = true;
    }
}

template<typename T>
void XLUpBlock2dWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                              std::string                          prefix,
                                              std::unordered_map<std::string, T*>& lora_weights,
                                              float                                lora_alpha,
                                              FtCudaDataType                       lora_file_type,
                                              cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    resnet_2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_2_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void XLUpBlock2dWeight<T>::loadLoraFromCache(std::string                          prefix,
                                             std::unordered_map<std::string, T*>& lora_weights,
                                             float                                lora_alpha,
                                             bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    resnet_2d_block_weight1->loadLoraFromCache(
        prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight2->loadLoraFromCache(
        prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight3->loadLoraFromCache(
        prefix + std::string("resnets_2_"), lora_weights, lora_alpha, from_outside);
}

template class XLUpBlock2dWeight<float>;
template class XLUpBlock2dWeight<half>;
}  // namespace lyradiff