#include "GLVControlNetModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
GLVControlNetModelWeight<T>::GLVControlNetModelWeight(const bool use_runtime_augemb, LyraQuantType quant_level)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (use_runtime_augemb_) {
        texttime_embedding_weight =
            TextTimeEmbeddingBlockWeight<T>(block_out_channels_[0], add_emb_input_dim_, temb_channels_, temb_channels_);
    }
    else {
        timestep_embedding_weight =
            TimestepEmbeddingBlockWeight<T>(block_out_channels_[0], temb_channels_, temb_channels_);
    }

    down_block_2d_weight =
        XLDownBlock2DWeight<T>(block_out_channels_[0], block_out_channels_[0], block_out_channels_[0], true);

    cross_attn_down_block_2d_weight1 = XLCrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                       block_out_channels_[1],
                                                                       temb_channels_,
                                                                       head_nums_[0],
                                                                       cross_attn_dim_,
                                                                       norm_num_groups_,
                                                                       inner_trans_nums_[0],
                                                                       true,
                                                                       quant_level);

    cross_attn_down_block_2d_weight2 = XLCrossAttnDownBlock2dWeight<T>(block_out_channels_[1],
                                                                       block_out_channels_[2],
                                                                       temb_channels_,
                                                                       head_nums_[1],
                                                                       cross_attn_dim_,
                                                                       norm_num_groups_,
                                                                       inner_trans_nums_[1],
                                                                       false,
                                                                       quant_level);

    // cout<<"midccc"<<in_channels_<<","<<out_channels_<<endl;
    mid_block_2d_weight = XLUNetMidBlock2DCrossAttnWeight<T>(block_out_channels_[2],
                                                             head_nums_[1],
                                                             norm_num_groups_,
                                                             cross_attn_dim_,
                                                             10,
                                                             block_out_channels_[2],
                                                             quant_level);
}

template<typename T>
GLVControlNetModelWeight<T>::~GLVControlNetModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);

        deviceFree(input_hint_conv_weight);
        deviceFree(input_hint_conv_bias);

        conv_in_weight = nullptr;
        conv_in_bias   = nullptr;

        input_hint_conv_weight = nullptr;
        input_hint_conv_bias   = nullptr;
    }
}

template<typename T>
void GLVControlNetModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&conv_in_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&conv_in_bias, block_out_channels_[0]);

        deviceMalloc(&input_hint_conv_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&input_hint_conv_bias, block_out_channels_[0]);

        if (use_runtime_augemb_) {
            texttime_embedding_weight.mallocWeights();
        }
        else {
            timestep_embedding_weight.mallocWeights();
        }
        down_block_2d_weight.mallocWeights();
        cross_attn_down_block_2d_weight1.mallocWeights();
        cross_attn_down_block_2d_weight2.mallocWeights();
        mid_block_2d_weight.mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void GLVControlNetModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
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

    loadWeightFromBin<T>(input_hint_conv_weight,
                         {input_channels_ * 3 * 3 * block_out_channels_[0]},
                         prefix + "controlnet_cond_embedding.conv_in.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(input_hint_conv_bias,
                         {block_out_channels_[0]},
                         prefix + "controlnet_cond_embedding.conv_in.bias.bin",
                         model_file_type);

    if (use_runtime_augemb_) {
        texttime_embedding_weight.loadWeights(prefix, model_file_type);
    }
    else {
        timestep_embedding_weight.loadWeights(prefix + "time_embedding.", model_file_type);
    }

    down_block_2d_weight.loadWeights(prefix + "down_blocks.0.", model_file_type);
    cross_attn_down_block_2d_weight1.loadWeights(prefix + "down_blocks.1.", model_file_type);
    cross_attn_down_block_2d_weight2.loadWeights(prefix + "down_blocks.2.", model_file_type);
    mid_block_2d_weight.loadWeights(prefix + "mid_block.", model_file_type);
}

template<typename T>
void GLVControlNetModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                       std::unordered_map<std::string, void*>& weights,
                                                       cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_conv_in_weight                   = weights[prefix + "conv_in.weight"];
    void* tmp_conv_in_bias                     = weights[prefix + "conv_in.bias"];
    void* tmp_controlnet_cond_embedding_weight = weights[prefix + "controlnet_cond_embedding.conv_in.weight"];
    void* tmp_controlnet_cond_embedding_bias   = weights[prefix + "controlnet_cond_embedding.conv_in.bias"];

    cudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);

    cudaMemcpy(input_hint_conv_weight,
               tmp_controlnet_cond_embedding_weight,
               sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0],
               memcpyKind);
    cudaMemcpy(
        input_hint_conv_bias, tmp_controlnet_cond_embedding_bias, sizeof(T) * block_out_channels_[0], memcpyKind);

    if (use_runtime_augemb_) {
        texttime_embedding_weight.loadWeightsFromCache(prefix, weights, memcpyKind);
    }
    else {
        timestep_embedding_weight.loadWeightsFromCache(prefix + "time_embedding.", weights, memcpyKind);
    }

    down_block_2d_weight.loadWeightsFromCache(prefix + "down_blocks.0.", weights, memcpyKind);
    cross_attn_down_block_2d_weight1.loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
    cross_attn_down_block_2d_weight2.loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);

    mid_block_2d_weight.loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);

    // cout << "all weights loaded" << endl;
}

template class GLVControlNetModelWeight<float>;
template class GLVControlNetModelWeight<half>;
}  // namespace lyradiff