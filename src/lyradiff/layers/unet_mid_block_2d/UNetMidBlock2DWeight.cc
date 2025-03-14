#include "UNetMidBlock2DWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {

template<typename T>
UNetMidBlock2DWeight<T>::UNetMidBlock2DWeight(const size_t in_channels,
                                              const size_t num_head,
                                              const size_t group_num,
                                              const size_t temb_channels):
    in_channels_(in_channels), num_head_(num_head), group_num_(group_num), temb_channels_(temb_channels)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    dim_per_head_ = in_channels_ / num_head_;

    attention_qkv_size_    = in_channels_ * in_channels_ * 3;
    attention_to_out_size_ = in_channels_ * in_channels_;

    resnet_0_weights_ = new Resnet2DBlockWeight<T>(in_channels_, in_channels_, temb_channels_ > 0);
    resnet_1_weights_ = new Resnet2DBlockWeight<T>(in_channels_, in_channels_, temb_channels_ > 0);
}

template<typename T>
UNetMidBlock2DWeight<T>::~UNetMidBlock2DWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    deviceFree(attention_qkv_weight);
    deviceFree(attention_qkv_bias);
    deviceFree(attention_to_out_weight);
    deviceFree(attention_to_out_bias);
    deviceFree(gnorm_gamma);
    deviceFree(gnorm_beta);

    attention_qkv_weight    = nullptr;
    attention_qkv_bias      = nullptr;
    attention_to_out_weight = nullptr;
    attention_to_out_bias   = nullptr;
    gnorm_gamma             = nullptr;
    gnorm_beta              = nullptr;
}

template<typename T>
void UNetMidBlock2DWeight<T>::mallocWeights()
{
    resnet_0_weights_->mallocWeights();
    resnet_1_weights_->mallocWeights();

    deviceMalloc(&attention_qkv_weight, attention_qkv_size_);
    deviceMalloc(&attention_qkv_bias, in_channels_ * 3);
    deviceMalloc(&attention_to_out_weight, attention_to_out_size_);
    deviceMalloc(&attention_to_out_bias, in_channels_);

    deviceMalloc(&gnorm_gamma, in_channels_);
    deviceMalloc(&gnorm_beta, in_channels_);

    is_maintain_buffer = true;
}

template<typename T>
void UNetMidBlock2DWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights_->loadWeights(resnet_0_model_prefix_, model_file_type);
    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights_->loadWeights(resnet_1_model_prefix_, model_file_type);

    int offset = 0;
    loadWeightFromBin<T>(&attention_qkv_weight[offset],
                         {in_channels_, in_channels_},
                         prefix + "attentions.0.to_q.weight.bin",
                         model_file_type);
    offset += in_channels_ * in_channels_;
    loadWeightFromBin<T>(&attention_qkv_weight[offset],
                         {in_channels_, in_channels_},
                         prefix + "attentions.0.to_k.weight.bin",
                         model_file_type);
    offset += in_channels_ * in_channels_;
    loadWeightFromBin<T>(&attention_qkv_weight[offset],
                         {in_channels_, in_channels_},
                         prefix + "attentions.0.to_v.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(attention_to_out_weight,
                         {in_channels_, in_channels_},
                         prefix + "attentions.0.to_out.0.weight.bin",
                         model_file_type);

    offset = 0;
    loadWeightFromBin<T>(
        &attention_qkv_bias[offset], {in_channels_}, prefix + "attentions.0.to_q.bias.bin", model_file_type);

    offset += in_channels_;
    loadWeightFromBin<T>(
        &attention_qkv_bias[offset], {in_channels_}, prefix + "attentions.0.to_k.bias.bin", model_file_type);

    offset += in_channels_;
    loadWeightFromBin<T>(
        &attention_qkv_bias[offset], {in_channels_}, prefix + "attentions.0.to_v.bias.bin", model_file_type);

    loadWeightFromBin<T>(
        attention_to_out_bias, {in_channels_}, prefix + "attentions.0.to_out.0.bias.bin", model_file_type);

    loadWeightFromBin<T>(gnorm_gamma, {in_channels_}, prefix + "attentions.0.group_norm.weight.bin", model_file_type);

    loadWeightFromBin<T>(gnorm_beta, {in_channels_}, prefix + "attentions.0.group_norm.bias.bin", model_file_type);
}

template<typename T>
void UNetMidBlock2DWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                   std::unordered_map<std::string, void*>& weights,
                                                   cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // attention1 weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    int offset = 0;

    void* tmp_attn_q      = weights[prefix + "attentions.0.to_q.weight"];
    void* tmp_attn_k      = weights[prefix + "attentions.0.to_k.weight"];
    void* tmp_attn_v      = weights[prefix + "attentions.0.to_v.weight"];
    void* tmp_attn_q_bias = weights[prefix + "attentions.0.to_q.bias"];
    void* tmp_attn_k_bias = weights[prefix + "attentions.0.to_k.bias"];
    void* tmp_attn_v_bias = weights[prefix + "attentions.0.to_v.bias"];

    void* tmp_attn_to_out      = weights[prefix + "attentions.0.to_out.0.weight"];
    void* tmp_attn_to_out_bias = weights[prefix + "attentions.0.to_out.0.bias"];

    void* tmp_norm_gamma = weights[prefix + "attentions.0.group_norm.weight"];
    void* tmp_norm_beta  = weights[prefix + "attentions.0.group_norm.bias"];

    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_weight[offset], tmp_attn_q, sizeof(T) * in_channels_ * in_channels_, memcpyKind);
    offset += in_channels_ * in_channels_;
    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_weight[offset], tmp_attn_k, sizeof(T) * in_channels_ * in_channels_, memcpyKind);
    offset += in_channels_ * in_channels_;
    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_weight[offset], tmp_attn_v, sizeof(T) * in_channels_ * in_channels_, memcpyKind);

    offset = 0;

    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_bias[offset], tmp_attn_q_bias, sizeof(T) * in_channels_, memcpyKind);
    offset += in_channels_;
    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_bias[offset], tmp_attn_k_bias, sizeof(T) * in_channels_, memcpyKind);
    offset += in_channels_;
    weight_loader_manager_glob->doCudaMemcpy(&attention_qkv_bias[offset], tmp_attn_v_bias, sizeof(T) * in_channels_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(attention_to_out_weight, tmp_attn_to_out, sizeof(T) * in_channels_ * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(attention_to_out_bias, tmp_attn_to_out_bias, sizeof(T) * in_channels_, memcpyKind);

    weight_loader_manager_glob->doCudaMemcpy(gnorm_gamma, tmp_norm_gamma, sizeof(T) * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(gnorm_beta, tmp_norm_beta, sizeof(T) * in_channels_, memcpyKind);

    std::string resnet_0_model_prefix_ = prefix + "resnets.0.";
    resnet_0_weights_->loadWeightsFromCache(resnet_0_model_prefix_, weights, memcpyKind);
    std::string resnet_1_model_prefix_ = prefix + "resnets.1.";
    resnet_1_weights_->loadWeightsFromCache(resnet_1_model_prefix_, weights, memcpyKind);
}

template class UNetMidBlock2DWeight<float>;
template class UNetMidBlock2DWeight<half>;
}  // namespace lyradiff