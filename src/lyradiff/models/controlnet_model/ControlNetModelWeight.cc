#include "ControlNetModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ControlNetModelWeight<T>::ControlNetModelWeight(const LyraQuantType quant_level)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // this->quant_level_ = quant_level;
    timestep_embedding_weight =
        new TimestepEmbeddingBlockWeight<T>(block_out_channels_[0], temb_channels_, temb_channels_, quant_level);

    cross_attn_down_block_2d_weight1 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                         block_out_channels_[0],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level);

    cross_attn_down_block_2d_weight2 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                         block_out_channels_[1],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level);

    cross_attn_down_block_2d_weight3 = new CrossAttnDownBlock2dWeight<T>(block_out_channels_[1],
                                                                         block_out_channels_[2],
                                                                         temb_channels_,
                                                                         head_num_,
                                                                         cross_attn_dim_,
                                                                         norm_num_groups_,
                                                                         quant_level);

    down_block_2d_weight =
        new DownBlock2DWeight<T>(block_out_channels_[2], block_out_channels_[3], block_out_channels_[3]);

    mid_block_2d_weight = new UNetMidBlock2DCrossAttnWeight<T>(
        block_out_channels_[3], head_num_, norm_num_groups_, cross_attn_dim_, block_out_channels_[3], quant_level);

    controlnet_conditioning_embedding_weight =
        new ControlNetConditioningEmbeddingWeight<T>(controlnet_condition_channels_, block_out_channels_[0]);

    controlnet_final_conv_weight =
        new ControlNetFinalConvWeight<T>({320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280});
}

template<typename T>
ControlNetModelWeight<T>::~ControlNetModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cout << "~ControlNetModelWeight()" << endl;
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);

        delete timestep_embedding_weight;
        delete cross_attn_down_block_2d_weight1;
        delete cross_attn_down_block_2d_weight2;
        delete cross_attn_down_block_2d_weight3;
        delete down_block_2d_weight;
        delete mid_block_2d_weight;
        delete controlnet_conditioning_embedding_weight;
        delete controlnet_final_conv_weight;

        conv_in_weight = nullptr;
        conv_in_bias   = nullptr;
    }
    cout << "~ControlNetModelWeight() end" << endl;
}

template<typename T>
void ControlNetModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&conv_in_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&conv_in_bias, block_out_channels_[0]);

        timestep_embedding_weight->mallocWeights();
        cross_attn_down_block_2d_weight1->mallocWeights();
        cross_attn_down_block_2d_weight2->mallocWeights();
        cross_attn_down_block_2d_weight3->mallocWeights();
        down_block_2d_weight->mallocWeights();
        mid_block_2d_weight->mallocWeights();
        controlnet_conditioning_embedding_weight->mallocWeights();
        controlnet_final_conv_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void ControlNetModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
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

    timestep_embedding_weight->loadWeights(prefix + "time_embedding.", model_file_type);
    cross_attn_down_block_2d_weight1->loadWeights(prefix + "down_blocks.0.", model_file_type);
    cross_attn_down_block_2d_weight2->loadWeights(prefix + "down_blocks.1.", model_file_type);
    cross_attn_down_block_2d_weight3->loadWeights(prefix + "down_blocks.2.", model_file_type);
    down_block_2d_weight->loadWeights(prefix + "down_blocks.3.", model_file_type);
    mid_block_2d_weight->loadWeights(prefix + "mid_block.", model_file_type);
    controlnet_conditioning_embedding_weight->loadWeights(prefix + "controlnet_cond_embedding.", model_file_type);
    controlnet_final_conv_weight->loadWeights(prefix, model_file_type);

    cout << "all weights loaded" << endl;
}

template<typename T>
void ControlNetModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                    std::unordered_map<std::string, void*>& weights,
                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_conv_in_weight = weights[prefix + "conv_in.weight"];
    void* tmp_conv_in_bias   = weights[prefix + "conv_in.bias"];

    cudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);

    timestep_embedding_weight->loadWeightsFromCache(prefix + "time_embedding.", weights, memcpyKind);
    cross_attn_down_block_2d_weight1->loadWeightsFromCache(prefix + "down_blocks.0.", weights, memcpyKind);
    cross_attn_down_block_2d_weight2->loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
    cross_attn_down_block_2d_weight3->loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);
    down_block_2d_weight->loadWeightsFromCache(prefix + "down_blocks.3.", weights, memcpyKind);
    mid_block_2d_weight->loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);
    controlnet_conditioning_embedding_weight->loadWeightsFromCache(
        prefix + "controlnet_cond_embedding.", weights, memcpyKind);
    controlnet_final_conv_weight->loadWeightsFromCache(prefix, weights, memcpyKind);

    // cout << "all weights loaded" << endl;
}

template class ControlNetModelWeight<float>;
template class ControlNetModelWeight<half>;
}  // namespace lyradiff