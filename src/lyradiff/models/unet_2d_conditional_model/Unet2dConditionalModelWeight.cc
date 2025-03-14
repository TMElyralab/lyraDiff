#include "Unet2dConditionalModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
Unet2dConditionalModelWeight<T>::Unet2dConditionalModelWeight(size_t              input_channels,
                                                              size_t              output_channels,
                                                              const LyraQuantType quant_level,
                                                              IAllocator*         allocator,
                                                              const std::string& sd_ver):
    input_channels_(input_channels), output_channels_(output_channels)
{
    if (sd_ver == "sd2")
        cross_attn_dim_ = 1024;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    this->quant_level_ = quant_level;
    this->allocator_   = allocator;

    timestep_embedding_weight = new TimestepEmbeddingBlockWeight<T>(
        block_out_channels_[0], temb_channels_, temb_channels_, quant_level, allocator);

    cross_attn_down_block_2d_weight1 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                         block_out_channels_[0],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level,
                                                                         allocator);

    cross_attn_down_block_2d_weight2 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                         block_out_channels_[1],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level,
                                                                         allocator);

    cross_attn_down_block_2d_weight3 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[1],
                                                                         block_out_channels_[2],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level,
                                                                         allocator);

    down_block_2d_weight =
        new DownBlock2DWeight<T>(block_out_channels_[2], block_out_channels_[3], block_out_channels_[3], allocator);

    mid_block_2d_weight = new UNetMidBlock2DCrossAttnWeight<T>(block_out_channels_[3],
                                                               head_num_,
                                                               norm_num_groups_,
                                                               cross_attn_dim_,
                                                               block_out_channels_[3],
                                                               quant_level,
                                                               allocator);

    up_block_2d_weight = new UpBlock2dWeight<T>(
        block_out_channels_[2], block_out_channels_[3], block_out_channels_[3], norm_num_groups_, allocator);

    cross_attn_up_block_2d_weight1 = new CrossAttnUpBlock2dWeight<T, true>(block_out_channels_[1],
                                                                           block_out_channels_[2],
                                                                           block_out_channels_[3],
                                                                           temb_channels_,
                                                                           head_num_,
                                                                           cross_attn_dim_,
                                                                           norm_num_groups_,
                                                                           quant_level,
                                                                           allocator);

    cross_attn_up_block_2d_weight2 = new CrossAttnUpBlock2dWeight<T, true>(block_out_channels_[0],
                                                                           block_out_channels_[1],
                                                                           block_out_channels_[2],
                                                                           temb_channels_,
                                                                           head_num_,
                                                                           cross_attn_dim_,
                                                                           norm_num_groups_,
                                                                           quant_level,
                                                                           allocator);

    cross_attn_up_block_2d_weight3 = new CrossAttnUpBlock2dWeight<T, false>(block_out_channels_[0],
                                                                            block_out_channels_[0],
                                                                            block_out_channels_[1],
                                                                            temb_channels_,
                                                                            head_num_,
                                                                            cross_attn_dim_,
                                                                            norm_num_groups_,
                                                                            quant_level,
                                                                            allocator);

    this->vec_basic_transformer_container_weights = {cross_attn_down_block_2d_weight1,
                                                     cross_attn_down_block_2d_weight2,
                                                     cross_attn_down_block_2d_weight3,
                                                     mid_block_2d_weight,
                                                     cross_attn_up_block_2d_weight1,
                                                     cross_attn_up_block_2d_weight2,
                                                     cross_attn_up_block_2d_weight3};

    this->lora_layer_map = {{"down_blocks_0", cross_attn_down_block_2d_weight1},
                            {"down_blocks_1", cross_attn_down_block_2d_weight2},
                            {"down_blocks_2", cross_attn_down_block_2d_weight3},
                            {"down_blocks_3", down_block_2d_weight},
                            {"mid_block", mid_block_2d_weight},
                            {"up_blocks_0", up_block_2d_weight},
                            {"up_blocks_1", cross_attn_up_block_2d_weight1},
                            {"up_blocks_2", cross_attn_up_block_2d_weight2},
                            {"up_blocks_3", cross_attn_up_block_2d_weight3}};
}

template<typename T>
Unet2dConditionalModelWeight<T>::~Unet2dConditionalModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);
        deviceFree(conv_out_norm_beta);
        deviceFree(conv_out_norm_gamma);
        deviceFree(conv_out_weight);
        deviceFree(conv_out_bias);

        conv_in_weight = nullptr;
        conv_in_bias   = nullptr;

        conv_out_norm_beta  = nullptr;
        conv_out_norm_gamma = nullptr;

        conv_out_weight = nullptr;
        conv_out_bias   = nullptr;

        delete timestep_embedding_weight;
        delete cross_attn_down_block_2d_weight1;
        delete cross_attn_down_block_2d_weight2;
        delete cross_attn_down_block_2d_weight3;
        delete down_block_2d_weight;
        delete mid_block_2d_weight;
        delete up_block_2d_weight;
        delete cross_attn_up_block_2d_weight1;
        delete cross_attn_up_block_2d_weight2;
        delete cross_attn_up_block_2d_weight3;
    }
}

template<typename T>
void Unet2dConditionalModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&conv_in_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&conv_in_bias, block_out_channels_[0]);
        deviceMalloc(&conv_out_norm_beta, block_out_channels_[0]);
        deviceMalloc(&conv_out_norm_gamma, block_out_channels_[0]);
        deviceMalloc(&conv_out_weight, block_out_channels_[0] * 3 * 3 * output_channels_);
        deviceMalloc(&conv_out_bias, output_channels_);

        timestep_embedding_weight->mallocWeights();
        cross_attn_down_block_2d_weight1->mallocWeights();
        cross_attn_down_block_2d_weight2->mallocWeights();
        cross_attn_down_block_2d_weight3->mallocWeights();
        down_block_2d_weight->mallocWeights();
        mid_block_2d_weight->mallocWeights();
        up_block_2d_weight->mallocWeights();
        cross_attn_up_block_2d_weight1->mallocWeights();
        cross_attn_up_block_2d_weight2->mallocWeights();
        cross_attn_up_block_2d_weight3->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void Unet2dConditionalModelWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                         std::string                          prefix,
                                                         std::unordered_map<std::string, T*>& lora_weights,
                                                         float                                lora_alpha,
                                                         FtCudaDataType                       lora_file_type,
                                                         cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cross_attn_down_block_2d_weight1->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);
    cross_attn_down_block_2d_weight2->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);
    cross_attn_down_block_2d_weight3->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);
    down_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_3_", lora_weights, lora_alpha, lora_file_type, stream);
    mid_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "mid_block_", lora_weights, lora_alpha, lora_file_type, stream);
    up_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);
    cross_attn_up_block_2d_weight1->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);
    cross_attn_up_block_2d_weight2->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);
    cross_attn_up_block_2d_weight3->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_3_", lora_weights, lora_alpha, lora_file_type, stream);
}

template<typename T>
void Unet2dConditionalModelWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                        std::unordered_map<std::string, T*>& lora_weights,
                                                        float                                lora_alpha,
                                                        bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cross_attn_down_block_2d_weight1->loadLoraFromCache(
        prefix + "down_blocks_0_", lora_weights, lora_alpha, from_outside);
    cross_attn_down_block_2d_weight2->loadLoraFromCache(
        prefix + "down_blocks_1_", lora_weights, lora_alpha, from_outside);
    cross_attn_down_block_2d_weight3->loadLoraFromCache(
        prefix + "down_blocks_2_", lora_weights, lora_alpha, from_outside);
    down_block_2d_weight->loadLoraFromCache(prefix + "down_blocks_3_", lora_weights, lora_alpha, from_outside);
    mid_block_2d_weight->loadLoraFromCache(prefix + "mid_block_", lora_weights, lora_alpha, from_outside);
    up_block_2d_weight->loadLoraFromCache(prefix + "up_blocks_0_", lora_weights, lora_alpha, from_outside);
    cross_attn_up_block_2d_weight1->loadLoraFromCache(prefix + "up_blocks_1_", lora_weights, lora_alpha, from_outside);
    cross_attn_up_block_2d_weight2->loadLoraFromCache(prefix + "up_blocks_2_", lora_weights, lora_alpha, from_outside);
    cross_attn_up_block_2d_weight3->loadLoraFromCache(prefix + "up_blocks_3_", lora_weights, lora_alpha, from_outside);
}

template<typename T>
void Unet2dConditionalModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    loadWeightFromBin<T>(conv_in_weight,
                         {input_channels_ * 3 * 3 * block_out_channels_[0]},
                         prefix + "conv_in.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_in_bias, {block_out_channels_[0]}, prefix + "conv_in.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        conv_out_norm_gamma, {block_out_channels_[0]}, prefix + "conv_norm_out.weight.bin", model_file_type);
    loadWeightFromBin<T>(
        conv_out_norm_beta, {block_out_channels_[0]}, prefix + "conv_norm_out.bias.bin", model_file_type);
    loadWeightFromBin<T>(conv_out_weight,
                         {block_out_channels_[0] * 3 * 3 * output_channels_},
                         prefix + "conv_out.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_out_bias, {output_channels_}, prefix + "conv_out.bias.bin", model_file_type);
    timestep_embedding_weight->loadWeights(prefix + "time_embedding.", model_file_type);
    cross_attn_down_block_2d_weight1->loadWeights(prefix + "down_blocks.0.", model_file_type);
    cross_attn_down_block_2d_weight2->loadWeights(prefix + "down_blocks.1.", model_file_type);
    cross_attn_down_block_2d_weight3->loadWeights(prefix + "down_blocks.2.", model_file_type);
    down_block_2d_weight->loadWeights(prefix + "down_blocks.3.", model_file_type);
    mid_block_2d_weight->loadWeights(prefix + "mid_block.", model_file_type);
    up_block_2d_weight->loadWeights(prefix + "up_blocks.0.", model_file_type);
    cross_attn_up_block_2d_weight1->loadWeights(prefix + "up_blocks.1.", model_file_type);
    cross_attn_up_block_2d_weight2->loadWeights(prefix + "up_blocks.2.", model_file_type);
    cross_attn_up_block_2d_weight3->loadWeights(prefix + "up_blocks.3.", model_file_type);
}

template<typename T>
void Unet2dConditionalModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                           std::unordered_map<std::string, void*>& weights,
                                                           cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_conv_in_weight      = weights[prefix + "conv_in.weight"];
    void* tmp_conv_in_bias        = weights[prefix + "conv_in.bias"];
    void* tmp_conv_out_norm_gamma = weights[prefix + "conv_norm_out.weight"];
    void* tmp_conv_out_norm_beta  = weights[prefix + "conv_norm_out.bias"];
    void* tmp_conv_out_weight     = weights[prefix + "conv_out.weight"];
    void* tmp_conv_out_bias       = weights[prefix + "conv_out.bias"];

    weight_loader_manager_glob->doCudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        conv_out_norm_gamma, tmp_conv_out_norm_gamma, sizeof(T) * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        conv_out_norm_beta, tmp_conv_out_norm_beta, sizeof(T) * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv_out_weight,
                                             tmp_conv_out_weight,
                                             sizeof(T) * block_out_channels_[0] * 3 * 3 * output_channels_,
                                             memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        conv_out_bias, tmp_conv_out_bias, sizeof(T) * output_channels_, memcpyKind);

    timestep_embedding_weight->loadWeightsFromCache(prefix + "time_embedding.", weights, memcpyKind);
    cross_attn_down_block_2d_weight1->loadWeightsFromCache(prefix + "down_blocks.0.", weights, memcpyKind);
    cross_attn_down_block_2d_weight2->loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
    cross_attn_down_block_2d_weight3->loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);
    down_block_2d_weight->loadWeightsFromCache(prefix + "down_blocks.3.", weights, memcpyKind);
    mid_block_2d_weight->loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);
    up_block_2d_weight->loadWeightsFromCache(prefix + "up_blocks.0.", weights, memcpyKind);
    cross_attn_up_block_2d_weight1->loadWeightsFromCache(prefix + "up_blocks.1.", weights, memcpyKind);
    cross_attn_up_block_2d_weight2->loadWeightsFromCache(prefix + "up_blocks.2.", weights, memcpyKind);
    cross_attn_up_block_2d_weight3->loadWeightsFromCache(prefix + "up_blocks.3.", weights, memcpyKind);
}

template<typename T>
void Unet2dConditionalModelWeight<T>::loadS3DiffLoraFromStateDict(std::unordered_map<std::string, T*>& lora_weights,
                                                                  bool                                 is_alpha)
{
    auto& lora_container = weight_loader_manager_glob->map_lora_container["unet"];
    for (auto iter : lora_weights) {
        lora_container.add_lora_weight(iter.first, iter.second, is_alpha);
    }
}

template class Unet2dConditionalModelWeight<float>;
template class Unet2dConditionalModelWeight<half>;
}  // namespace lyradiff