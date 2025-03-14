#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {
template<typename T, bool ADD_UPSAMPLE>
CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::CrossAttnUpBlock2dWeight(const size_t        in_channels,
                                                                    const size_t        out_channels,
                                                                    const size_t        prev_output_channel,
                                                                    const size_t        temb_channels,
                                                                    const size_t        head_num,
                                                                    const size_t        cross_attn_dim,
                                                                    const size_t        norm_num_groups,
                                                                    const LyraQuantType quant_level,
                                                                    IAllocator*         allocator):
    in_channels_(in_channels),
    out_channels_(out_channels),
    prev_output_channel_(prev_output_channel),
    temb_channels_(temb_channels),
    head_num_(head_num),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->quant_level_ = quant_level;
    this->allocator_   = allocator;

    transformer2d_block_weight1 = new Transformer2dBlockWeight<T>(
        out_channels_, head_num_, out_channels_ / head_num_, cross_attn_dim_, norm_num_groups_, quant_level, allocator);

    transformer2d_block_weight2 = new Transformer2dBlockWeight<T>(
        out_channels_, head_num_, out_channels_ / head_num_, cross_attn_dim_, norm_num_groups_, quant_level, allocator);

    transformer2d_block_weight3 = new Transformer2dBlockWeight<T>(
        out_channels_, head_num_, out_channels_ / head_num_, cross_attn_dim_, norm_num_groups_, quant_level, allocator);

    resnet_2d_block_weight1 =
        new Resnet2DBlockWeight<T>(out_channels_ + prev_output_channel_, out_channels_, true, allocator);
    resnet_2d_block_weight2 = new Resnet2DBlockWeight<T>(out_channels_ + out_channels_, out_channels_, true, allocator);
    resnet_2d_block_weight3 = new Resnet2DBlockWeight<T>(out_channels_ + in_channels_, out_channels_, true, allocator);

    upsampler_weight_size_ = out_channels_ * 3 * 3 * out_channels_;

    this->vec_basic_transformer_container_weights = {
        transformer2d_block_weight1, transformer2d_block_weight2, transformer2d_block_weight3};
    // TODO: 加入 resnet 2d block weights

    this->lora_layer_map = {{"attentions_0", transformer2d_block_weight1},
                            {"attentions_1", transformer2d_block_weight2},
                            {"attentions_2", transformer2d_block_weight3},
                            {"resnets_0", resnet_2d_block_weight1},
                            {"resnets_1", resnet_2d_block_weight2},
                            {"resnets_2", resnet_2d_block_weight3}};
}

template<typename T, bool ADD_UPSAMPLE>
CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::~CrossAttnUpBlock2dWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        if (ADD_UPSAMPLE) {
            deviceFree(upsampler_weight);
            deviceFree(upsampler_bias);

            upsampler_weight = nullptr;
            upsampler_bias   = nullptr;
        }

        delete transformer2d_block_weight1;
        delete transformer2d_block_weight2;
        delete transformer2d_block_weight3;
        delete resnet_2d_block_weight1;
        delete resnet_2d_block_weight2;
        delete resnet_2d_block_weight3;

        transformer2d_block_weight1 = nullptr;
        transformer2d_block_weight2 = nullptr;
        transformer2d_block_weight3 = nullptr;
        resnet_2d_block_weight1     = nullptr;
        resnet_2d_block_weight2     = nullptr;
        resnet_2d_block_weight3     = nullptr;
    }
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (ADD_UPSAMPLE) {
        deviceMalloc(&upsampler_weight, upsampler_weight_size_);
        deviceMalloc(&upsampler_bias, out_channels_);

        this->lora_weight_map = {
            {"upsamplers_0_conv", new LoraWeight<T>({out_channels_, 3, 3, out_channels_}, upsampler_weight)}};
    }
    transformer2d_block_weight1->mallocWeights();
    transformer2d_block_weight2->mallocWeights();
    transformer2d_block_weight3->mallocWeights();
    resnet_2d_block_weight1->mallocWeights();
    resnet_2d_block_weight2->mallocWeights();
    resnet_2d_block_weight3->mallocWeights();

    is_maintain_buffer = true;
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    if (ADD_UPSAMPLE) {
        loadWeightFromBin<T>(
            upsampler_weight, {upsampler_weight_size_}, prefix + "upsamplers.0.conv.weight.bin", model_file_type);
        loadWeightFromBin<T>(upsampler_bias, {out_channels_}, prefix + "upsamplers.0.conv.bias.bin", model_file_type);
    }

    transformer2d_block_weight1->loadWeights(prefix + std::string("attentions.0."), model_file_type);
    transformer2d_block_weight2->loadWeights(prefix + std::string("attentions.1."), model_file_type);
    transformer2d_block_weight3->loadWeights(prefix + std::string("attentions.2."), model_file_type);

    resnet_2d_block_weight1->loadWeights(prefix + std::string("resnets.0."), model_file_type);
    resnet_2d_block_weight2->loadWeights(prefix + std::string("resnets.1."), model_file_type);
    resnet_2d_block_weight3->loadWeights(prefix + std::string("resnets.2."), model_file_type);
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::loadWeightsFromCache(std::string                             prefix,
                                                                     std::unordered_map<std::string, void*>& weights,
                                                                     cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // norm weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    if (ADD_UPSAMPLE) {
        void* tmp_upsampler_weight = weights[prefix + "upsamplers.0.conv.weight"];
        void* tmp_upsampler_bias   = weights[prefix + "upsamplers.0.conv.bias"];

        weight_loader_manager_glob->doCudaMemcpy(upsampler_weight, tmp_upsampler_weight, sizeof(T) * upsampler_weight_size_, memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(upsampler_bias, tmp_upsampler_bias, sizeof(T) * out_channels_, memcpyKind);
    }

    transformer2d_block_weight1->loadWeightsFromCache(prefix + std::string("attentions.0."), weights, memcpyKind);
    transformer2d_block_weight2->loadWeightsFromCache(prefix + std::string("attentions.1."), weights, memcpyKind);
    transformer2d_block_weight3->loadWeightsFromCache(prefix + std::string("attentions.2."), weights, memcpyKind);
    resnet_2d_block_weight1->loadWeightsFromCache(prefix + std::string("resnets.0."), weights, memcpyKind);
    resnet_2d_block_weight2->loadWeightsFromCache(prefix + std::string("resnets.1."), weights, memcpyKind);
    resnet_2d_block_weight3->loadWeightsFromCache(prefix + std::string("resnets.2."), weights, memcpyKind);
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&upsampler_weight_lora_buf_, upsampler_weight_size_);

        is_maintain_lora = true;
    }
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::loadLoraFromWeight(std::string                          lora_path,
                                                                   std::string                          prefix,
                                                                   std::unordered_map<std::string, T*>& lora_weights,
                                                                   float                                lora_alpha,
                                                                   FtCudaDataType                       lora_file_type,
                                                                   cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (ADD_UPSAMPLE && checkIfFileExist(lora_path + prefix + "upsamplers_0_conv.bin")) {

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

    transformer2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_0_"), lora_weights, lora_alpha, lora_file_type, stream);
    transformer2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_1_"), lora_weights, lora_alpha, lora_file_type, stream);
    transformer2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("attentions_2_"), lora_weights, lora_alpha, lora_file_type, stream);

    resnet_2d_block_weight1->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_0_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight2->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_1_"), lora_weights, lora_alpha, lora_file_type);
    resnet_2d_block_weight3->loadLoraFromWeight(
        lora_path, prefix + std::string("resnets_2_"), lora_weights, lora_alpha, lora_file_type);
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>::loadLoraFromCache(std::string                          prefix,
                                                                  std::unordered_map<std::string, T*>& lora_weights,
                                                                  float                                lora_alpha,
                                                                  bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (ADD_UPSAMPLE && lora_weights.find(prefix + "upsamplers_0_conv") != lora_weights.end()) {

        if (!is_maintain_lora) {
            mallocLoraBuffer();
        }

        T* upsamplers_0_conv = lora_weights[prefix + "upsamplers_0_conv"];
        cudaMemcpy(
            upsampler_weight_lora_buf_, upsamplers_0_conv, sizeof(T) * upsampler_weight_size_, cudaMemcpyHostToDevice);
        invokeLoadLora<T>(upsampler_weight, upsampler_weight_lora_buf_, upsampler_weight_size_, lora_alpha);
    }

    transformer2d_block_weight1->loadLoraFromCache(
        prefix + std::string("attentions_0_"), lora_weights, lora_alpha, from_outside);
    transformer2d_block_weight2->loadLoraFromCache(
        prefix + std::string("attentions_1_"), lora_weights, lora_alpha, from_outside);
    transformer2d_block_weight3->loadLoraFromCache(
        prefix + std::string("attentions_2_"), lora_weights, lora_alpha, from_outside);

    resnet_2d_block_weight1->loadLoraFromCache(
        prefix + std::string("resnets_0_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight2->loadLoraFromCache(
        prefix + std::string("resnets_1_"), lora_weights, lora_alpha, from_outside);
    resnet_2d_block_weight3->loadLoraFromCache(
        prefix + std::string("resnets_2_"), lora_weights, lora_alpha, from_outside);
}

template class CrossAttnUpBlock2dWeight<float, true>;
template class CrossAttnUpBlock2dWeight<float, false>;
template class CrossAttnUpBlock2dWeight<half, true>;
template class CrossAttnUpBlock2dWeight<half, false>;
}  // namespace lyradiff