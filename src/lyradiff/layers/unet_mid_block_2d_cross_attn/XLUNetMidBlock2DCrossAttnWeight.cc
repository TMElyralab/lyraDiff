#include "XLUNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLUNetMidBlock2DCrossAttnWeight<T>::XLUNetMidBlock2DCrossAttnWeight(const size_t  in_channels,
                                                                    const size_t  num_head,
                                                                    const size_t  group_num,
                                                                    const size_t  encoder_hidden_dim,
                                                                    const size_t  inner_trans_num,
                                                                    const size_t  out_channels,
                                                                    LyraQuantType quant_level,
                                                                    IAllocator*   allocator):
    in_channels_(in_channels),
    num_head_(num_head),
    group_num_(group_num),
    encoder_hidden_dim_(encoder_hidden_dim),
    inner_trans_num_(inner_trans_num),
    out_channels_(out_channels)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    dim_per_head_ = in_channels_ / num_head_;

    resnet_0_weights_ = new Resnet2DBlockWeight<T>(in_channels_, out_channels_, true, allocator);
    resnet_1_weights_ = new Resnet2DBlockWeight<T>(out_channels, out_channels_, true, allocator);

    attn_weights_ = new XLTransformer2dBlockWeight<T>(out_channels_,
                                                      num_head_,
                                                      dim_per_head_,
                                                      encoder_hidden_dim_,
                                                      group_num_,
                                                      inner_trans_num_,
                                                      quant_level,
                                                      allocator);

    this->vec_basic_transformer_container_weights = {attn_weights_};

    this->lora_layer_map = {
        {"attentions_0", attn_weights_}, {"resnets_0", resnet_0_weights_}, {"resnets_1", resnet_1_weights_}};
}

// template<typename T>
// XLUNetMidBlock2DCrossAttnWeight<T>::XLUNetMidBlock2DCrossAttnWeight(const size_t in_channels,
//                                                                     const size_t num_head,
//                                                                     const size_t group_num,
//                                                                     const size_t encoder_hidden_dim,
//                                                                     const size_t out_channels,
//                                                                     size_t       quant_level):
//     XLUNetMidBlock2DCrossAttnWeight(in_channels, num_head, group_num, encoder_hidden_dim, 10, out_channels,
//     quant_level)
// {
// }

template<typename T>
XLUNetMidBlock2DCrossAttnWeight<T>::~XLUNetMidBlock2DCrossAttnWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        MACROFreeBuffByDeviceMalloc(attn_weights_);
        MACROFreeBuffByDeviceMalloc(resnet_0_weights_);
        MACROFreeBuffByDeviceMalloc(resnet_1_weights_);
        is_maintain_buffer = false;
    }
    // delete attn_weights_;
}

template<typename T>
void XLUNetMidBlock2DCrossAttnWeight<T>::mallocWeights()
{
    resnet_0_weights_->mallocWeights();
    attn_weights_->mallocWeights();
    resnet_1_weights_->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T>
void XLUNetMidBlock2DCrossAttnWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights_->loadWeights(resnet_0_model_prefix_, model_file_type);

    std::string attn_model_prefix_ = prefix + "attentions.0.";
    attn_weights_->loadWeights(attn_model_prefix_, model_file_type);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights_->loadWeights(resnet_1_model_prefix_, model_file_type);
}

template<typename T>
void XLUNetMidBlock2DCrossAttnWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                              std::unordered_map<std::string, void*>& weights,
                                                              cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights_->loadWeightsFromCache(resnet_0_model_prefix_, weights, memcpyKind);

    std::string attn_model_prefix_ = prefix + "attentions.0.";
    attn_weights_->loadWeightsFromCache(attn_model_prefix_, weights, memcpyKind);

    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights_->loadWeightsFromCache(resnet_1_model_prefix_, weights, memcpyKind);
}

template<typename T>
void XLUNetMidBlock2DCrossAttnWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                            std::string                          prefix,
                                                            std::unordered_map<std::string, T*>& lora_weights,
                                                            float                                lora_alpha,
                                                            FtCudaDataType                       lora_file_type,
                                                            cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::string attn_model_prefix_ = prefix + "attentions_0_";
    attn_weights_->loadLoraFromWeight(lora_path, attn_model_prefix_, lora_weights, lora_alpha, lora_file_type, stream);
    resnet_0_weights_->loadLoraFromWeight(lora_path, prefix + "resnets_0_", lora_weights, lora_alpha, lora_file_type);
    resnet_1_weights_->loadLoraFromWeight(lora_path, prefix + "resnets_1_", lora_weights, lora_alpha, lora_file_type);
}

template<typename T>
void XLUNetMidBlock2DCrossAttnWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                           std::unordered_map<std::string, T*>& lora_weights,
                                                           float                                lora_alpha,
                                                           bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::string attn_model_prefix_ = prefix + "attentions_0_";
    attn_weights_->loadLoraFromCache(attn_model_prefix_, lora_weights, lora_alpha, from_outside);
    resnet_0_weights_->loadLoraFromCache(prefix + "resnets_0_", lora_weights, lora_alpha, from_outside);
    resnet_1_weights_->loadLoraFromCache(prefix + "resnets_1_", lora_weights, lora_alpha, from_outside);
}

template class XLUNetMidBlock2DCrossAttnWeight<float>;
template class XLUNetMidBlock2DCrossAttnWeight<half>;
}  // namespace lyradiff