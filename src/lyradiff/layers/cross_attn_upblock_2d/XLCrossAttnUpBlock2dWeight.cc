#include "src/lyradiff/layers/cross_attn_upblock_2d/XLCrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {
template<typename T>
XLCrossAttnUpBlock2dWeight<T>::XLCrossAttnUpBlock2dWeight(const size_t  in_channels,
                                                          const size_t  out_channels,
                                                          const size_t  prev_output_channel,
                                                          const size_t  temb_channels,
                                                          const size_t  head_num,
                                                          const size_t  cross_attn_dim,
                                                          const size_t  norm_num_groups,
                                                          const size_t  inner_trans_num,
                                                          LyraQuantType quant_level,
                                                          IAllocator*   allocator):
    in_channels_(in_channels),
    out_channels_(out_channels),
    prev_output_channel_(prev_output_channel),
    temb_channels_(temb_channels),
    head_num_(head_num),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim),
    inner_trans_num_(inner_trans_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_ = allocator;

    xltransformer2d_block_weight1 = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                                      head_num_,
                                                                      out_channels_ / head_num_,
                                                                      cross_attn_dim_,
                                                                      norm_num_groups_,
                                                                      inner_trans_num_,
                                                                      quant_level,
                                                                      allocator);

    xltransformer2d_block_weight2 = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                                      head_num_,
                                                                      out_channels_ / head_num_,
                                                                      cross_attn_dim_,
                                                                      norm_num_groups_,
                                                                      inner_trans_num_,
                                                                      quant_level,
                                                                      allocator);

    xltransformer2d_block_weight3 = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                                      head_num_,
                                                                      out_channels_ / head_num_,
                                                                      cross_attn_dim_,
                                                                      norm_num_groups_,
                                                                      inner_trans_num_,
                                                                      quant_level,
                                                                      allocator);

    resnet_2d_block_weight1 =
        new Resnet2DBlockWeight<T>(out_channels_ + prev_output_channel_, out_channels_, true, allocator);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_ + out_channels_, out_channels_, true, allocator);
    resnet_2d_block_weight3 = new Resnet2DBlockWeight<T>(out_channels_ + in_channels_, out_channels_, true, allocator);

    upsampler_weight_size_ = out_channels_ * 3 * 3 * out_channels_;

    this->vec_basic_transformer_container_weights = {
        xltransformer2d_block_weight1, xltransformer2d_block_weight2, xltransformer2d_block_weight3};
    // TODO: 加入 resnet 2d block weights
    this->lora_layer_map = {{"attentions_0", xltransformer2d_block_weight1},
                            {"attentions_1", xltransformer2d_block_weight2},
                            {"attentions_2", xltransformer2d_block_weight3},
                            {"resnets_0", resnet_2d_block_weight1},
                            {"resnets_1", resnet_2d_block_weight2},
                            {"resnets_2", resnet_2d_block_weight3}};
}

template<typename T>
XLCrossAttnUpBlock2dWeight<T>::~XLCrossAttnUpBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(upsampler_weight);
        deviceFree(upsampler_bias);

        upsampler_weight = nullptr;
        upsampler_bias   = nullptr;

        delete xltransformer2d_block_weight1;
        delete xltransformer2d_block_weight2;
        delete xltransformer2d_block_weight3;
        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;
        delete resnet_2d_block_weight3;

        xltransformer2d_block_weight1 = nullptr;
        xltransformer2d_block_weight2 = nullptr;
        xltransformer2d_block_weight3 = nullptr;
        resnet_2d_block_weight1       = nullptr;
        resnet_2d_block_weight2       = nullptr;
        resnet_2d_block_weight3       = nullptr;
    }
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    deviceMalloc(&upsampler_weight, upsampler_weight_size_);
    deviceMalloc(&upsampler_bias, out_channels_);

    this->lora_weight_map = {
        {"upsamplers_0_conv", new LoraWeight<T>({out_channels_, 3, 3, out_channels_}, upsampler_weight)}};

    xltransformer2d_block_weight1->mallocWeights();
    xltransformer2d_block_weight2->mallocWeights();
    xltransformer2d_block_weight3->mallocWeights();
    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();
    resnet_2d_block_weight3->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    loadWeightFromBin<T>(
        upsampler_weight, {upsampler_weight_size_}, prefix + "upsamplers.0.conv.weight.bin", model_file_type);
    loadWeightFromBin<T>(upsampler_bias, {out_channels_}, prefix + "upsamplers.0.conv.bias.bin", model_file_type);

    xltransformer2d_block_weight1->loadWeights(prefix + std::string("attentions.0."), model_file_type);
    xltransformer2d_block_weight2->loadWeights(prefix + std::string("attentions.1."), model_file_type);
    xltransformer2d_block_weight3->loadWeights(prefix + std::string("attentions.2."), model_file_type);

    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
    resnet_2d_block_weight3->loadWeights(prefix + std::string("resnets.2."), model_file_type);
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                         std::unordered_map<std::string, void*>& weights,
                                                         cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_upsampler_weight = weights[prefix + "upsamplers.0.conv.weight"];
    void* tmp_upsampler_bias   = weights[prefix + "upsamplers.0.conv.bias"];

    cudaMemcpy(upsampler_weight, tmp_upsampler_weight, sizeof(T) * upsampler_weight_size_, memcpyKind);
    cudaMemcpy(upsampler_bias, tmp_upsampler_bias, sizeof(T) * out_channels_, memcpyKind);

    xltransformer2d_block_weight1->loadWeightsFromCache(prefix + std::string("attentions.0."), weights, memcpyKind);
    xltransformer2d_block_weight2->loadWeightsFromCache(prefix + std::string("attentions.1."), weights, memcpyKind);
    xltransformer2d_block_weight3->loadWeightsFromCache(prefix + std::string("attentions.2."), weights, memcpyKind);
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
    resnet_2d_block_weight3->loadWeightsFromCache(prefix + std::string("resnets.2."), weights, memcpyKind);
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&upsampler_weight_lora_buf_, upsampler_weight_size_);

        is_maintain_lora = true;
    }
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                       std::string                          prefix,
                                                       std::unordered_map<std::string, T*>& lora_weights,
                                                       float                                lora_alpha,
                                                       FtCudaDataType                       lora_file_type,
                                                       cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
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

    xltransformer2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_0_"), lora_weights, lora_alpha, lora_file_type, stream);
    xltransformer2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_1_"), lora_weights, lora_alpha, lora_file_type, stream);
    xltransformer2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_2_"), lora_weights, lora_alpha, lora_file_type, stream);

    resnet_2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_2_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void XLCrossAttnUpBlock2dWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                      std::unordered_map<std::string, T*>& lora_weights,
                                                      float                                lora_alpha,
                                                      bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (lora_weights.find(prefix + "upsamplers_0_conv") != lora_weights.end()) {

        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        T* upsamplers_0_conv = lora_weights[prefix + "upsamplers_0_conv"];
        cudaMemcpy(
            upsampler_weight_lora_buf_, upsamplers_0_conv, sizeof(T) * upsampler_weight_size_, cudaMemcpyHostToDevice);
        invokeLoadLora<T>(upsampler_weight, upsampler_weight_lora_buf_, upsampler_weight_size_, lora_alpha);
    }

    xltransformer2d_block_weight1->loadLoraFromCache(
        prefix + std::string("attentions_0_"), lora_weights, lora_alpha, from_outside);
    xltransformer2d_block_weight2->loadLoraFromCache(
        prefix + std::string("attentions_1_"), lora_weights, lora_alpha, from_outside);
    xltransformer2d_block_weight3->loadLoraFromCache(
        prefix + std::string("attentions_2_"), lora_weights, lora_alpha, from_outside);

    resnet_2d_block_weight1->loadLoraFromCache(
        prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight2->loadLoraFromCache(
        prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight3->loadLoraFromCache(
        prefix + std::string("resnets_2_"), lora_weights, lora_alpha, from_outside);
}

template class XLCrossAttnUpBlock2dWeight<float>;
template class XLCrossAttnUpBlock2dWeight<half>;
}  // namespace lyradiff