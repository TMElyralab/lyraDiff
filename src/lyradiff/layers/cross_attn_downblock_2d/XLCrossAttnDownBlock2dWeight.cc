#include "src/lyradiff/layers/cross_attn_downblock_2d/XLCrossAttnDownBlock2dWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLCrossAttnDownBlock2dWeight<T>::XLCrossAttnDownBlock2dWeight(const size_t  in_channels,
                                                              const size_t  out_channels,
                                                              const size_t  temb_channels,
                                                              const size_t  head_num,
                                                              const size_t  cross_attn_dim,
                                                              const size_t  norm_num_groups,
                                                              const size_t  inner_transblock_num,
                                                              bool          is_downsampler,
                                                              LyraQuantType quant_level,
                                                              IAllocator*   allocator):
    in_channels_(in_channels),
    out_channels_(out_channels),
    temb_channels_(temb_channels),
    head_num_(head_num),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim),
    inner_transblock_num_(inner_transblock_num),
    is_downsampler_(is_downsampler)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    xltransformer2d_block_weight1 = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                                      head_num_,
                                                                      out_channels_ / head_num_,
                                                                      cross_attn_dim_,
                                                                      norm_num_groups_,
                                                                      inner_transblock_num_,
                                                                      quant_level,
                                                                      allocator);

    xltransformer2d_block_weight2 = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                                      head_num_,
                                                                      out_channels_ / head_num_,
                                                                      cross_attn_dim_,
                                                                      norm_num_groups_,
                                                                      inner_transblock_num_,
                                                                      quant_level,
                                                                      allocator);

    resnet_2d_block_weight1 = new Resnet2DBlockWeight<T>(in_channels_, out_channels_, true, allocator);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_, out_channels_, true, allocator);

    this->quant_level_ = quant_level;
    this->allocator_   = allocator;
    // TODO: 加入 resnet 2d block weights
    this->vec_basic_transformer_container_weights = {xltransformer2d_block_weight1, xltransformer2d_block_weight2};

    this->lora_layer_map = {{"attentions_0", xltransformer2d_block_weight1},
                            {"attentions_1", xltransformer2d_block_weight2},
                            {"resnets_0", resnet_2d_block_weight1},
                            {"resnets_1", resnet_2d_block_weight2}};

    downsampler_weight_size_ = out_channels_ * 3 * 3 * out_channels_;
}

template<typename T>
XLCrossAttnDownBlock2dWeight<T>::~XLCrossAttnDownBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(downsampler_weight);
        deviceFree(downsampler_bias);

        downsampler_weight = nullptr;
        downsampler_bias   = nullptr;

        delete xltransformer2d_block_weight1;
        delete xltransformer2d_block_weight2;
        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;

        xltransformer2d_block_weight1 = nullptr;
        xltransformer2d_block_weight2 = nullptr;
        resnet_2d_block_weight1       = nullptr;
        resnet_2d_block_weight2       = nullptr;
    }
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_downsampler_) {
        deviceMalloc(&downsampler_weight, downsampler_weight_size_);
        deviceMalloc(&downsampler_bias, out_channels_);

        this->lora_weight_map = {
            {"downsamplers_0_conv", new LoraWeight<T>({out_channels_, 3, 3, out_channels_}, downsampler_weight)}};
    }

    xltransformer2d_block_weight1->mallocWeights();
    xltransformer2d_block_weight2->mallocWeights();
    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&downsampler_weight_lora_buf_, downsampler_weight_size_);

        is_maintain_lora = true;
    }
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    if (is_downsampler_) {
        // norm weights
        loadWeightFromBin<T>(downsampler_weight,
                             {out_channels_ * 3 * 3 * out_channels_},
                             prefix + "downsamplers.0.conv.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(
            downsampler_bias, {out_channels_}, prefix + "downsamplers.0.conv.bias.bin", model_file_type);
    }

    xltransformer2d_block_weight1->loadWeights(prefix + std::string("attentions.0."), model_file_type);
    xltransformer2d_block_weight2->loadWeights(prefix + std::string("attentions.1."), model_file_type);
    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                           std::unordered_map<std::string, void*>& weights,
                                                           cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
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

    xltransformer2d_block_weight1->loadWeightsFromCache(prefix + std::string("attentions.0."), weights, memcpyKind);
    xltransformer2d_block_weight2->loadWeightsFromCache(prefix + std::string("attentions.1."), weights, memcpyKind);
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                         std::string                          prefix,
                                                         std::unordered_map<std::string, T*>& lora_weights,
                                                         float                                lora_alpha,
                                                         FtCudaDataType                       lora_file_type,
                                                         cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // cout << "load lcm lora cross attn down block 2d" << endl;

    // for lcm lora
    if (checkIfFileExist(lora_path + prefix + "downsamplers_0_conv.bin")) {
        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        loadWeightFromBin<T>(downsampler_weight_lora_buf_,
                             {downsampler_weight_size_},
                             lora_path + prefix + "downsamplers_0_conv.bin",
                             lora_file_type);

        T* downsamplers_0_conv = (T*)malloc(sizeof(T) * downsampler_weight_size_);
        cudaMemcpy(downsamplers_0_conv,
                   downsampler_weight_lora_buf_,
                   sizeof(T) * downsampler_weight_size_,
                   cudaMemcpyDeviceToHost);
        lora_weights[prefix + "downsamplers_0_conv"] = downsamplers_0_conv;
        invokeLoadLora<T>(downsampler_weight, downsampler_weight_lora_buf_, downsampler_weight_size_, lora_alpha);
    }

    xltransformer2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_0_"), lora_weights, lora_alpha, lora_file_type, stream);
    xltransformer2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_1_"), lora_weights, lora_alpha, lora_file_type, stream);

    resnet_2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void XLCrossAttnDownBlock2dWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                        std::unordered_map<std::string, T*>& lora_weights,
                                                        float                                lora_alpha,
                                                        bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // for lcm lora
    if (lora_weights.find(prefix + "downsamplers_0_conv") != lora_weights.end()) {

        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        T* downsamplers_0_conv = lora_weights[prefix + "downsamplers_0_conv"];
        cudaMemcpy(downsampler_weight_lora_buf_,
                   downsamplers_0_conv,
                   sizeof(T) * downsampler_weight_size_,
                   cudaMemcpyHostToDevice);
        invokeLoadLora<T>(downsampler_weight, downsampler_weight_lora_buf_, downsampler_weight_size_, lora_alpha);
    }

    xltransformer2d_block_weight1->loadLoraFromCache(
        prefix + std::string("attentions_0_"), lora_weights, lora_alpha, from_outside);
    xltransformer2d_block_weight2->loadLoraFromCache(
        prefix + std::string("attentions_1_"), lora_weights, lora_alpha, from_outside);

    resnet_2d_block_weight1->loadLoraFromCache(
        prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight2->loadLoraFromCache(
        prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
}
template class XLCrossAttnDownBlock2dWeight<float>;
template class XLCrossAttnDownBlock2dWeight<half>;
}  // namespace lyradiff