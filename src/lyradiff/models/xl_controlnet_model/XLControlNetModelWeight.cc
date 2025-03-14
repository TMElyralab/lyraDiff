#include "XLControlNetModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLControlNetModelWeight<T>::XLControlNetModelWeight(const std::string controlnet_mode,
                                                    const bool        use_runtime_augemb,
                                                    LyraQuantType     quant_level):
    controlnet_mode_(controlnet_mode), use_runtime_augemb_(use_runtime_augemb)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (use_runtime_augemb_) {
        texttime_embedding_weight = new TextTimeEmbeddingBlockWeight<T>(
            block_out_channels_[0], add_emb_input_dim_, temb_channels_, temb_channels_);
    }
    else {
        timestep_embedding_weight =
            new TimestepEmbeddingBlockWeight<T>(block_out_channels_[0], temb_channels_, temb_channels_);
    }

    down_block_2d_weight =
        new XLDownBlock2DWeight<T>(block_out_channels_[0], block_out_channels_[0], block_out_channels_[0], true);

    if (controlnet_mode_ == "large") {
        cross_attn_down_block_2d_weight1 = new XLCrossAttnDownBlock2dWeight<T>(block_out_channels_[0],
                                                                               block_out_channels_[1],
                                                                               temb_channels_,
                                                                               head_nums_[0],
                                                                               cross_attn_dim_,
                                                                               norm_num_groups_,
                                                                               inner_trans_nums_[0],
                                                                               true,
                                                                               quant_level);

        cross_attn_down_block_2d_weight2 = new XLCrossAttnDownBlock2dWeight<T>(block_out_channels_[1],
                                                                               block_out_channels_[2],
                                                                               temb_channels_,
                                                                               head_nums_[1],
                                                                               cross_attn_dim_,
                                                                               norm_num_groups_,
                                                                               inner_trans_nums_[1],
                                                                               false,
                                                                               quant_level);

        // cout<<"midccc"<<in_channels_<<","<<out_channels_<<endl;
        mid_block_2d_weight = new XLUNetMidBlock2DCrossAttnWeight<T>(block_out_channels_[2],
                                                                     head_nums_[1],
                                                                     norm_num_groups_,
                                                                     cross_attn_dim_,
                                                                     10,
                                                                     block_out_channels_[2],
                                                                     quant_level);
    }

    else if (controlnet_mode_ == "small") {
        small_down_block_2d_weight1 =
            new XLDownBlock2DWeight<T>(block_out_channels_[0], block_out_channels_[1], block_out_channels_[1], true);

        small_down_block_2d_weight2 =
            new XLDownBlock2DWeight<T>(block_out_channels_[1], block_out_channels_[2], block_out_channels_[2], false);

        small_mid_block_2d_weight = new XLUNetMidBlock2DCrossAttnWeight<T>(block_out_channels_[2],
                                                                           head_nums_[1],
                                                                           norm_num_groups_,
                                                                           cross_attn_dim_,
                                                                           0,
                                                                           block_out_channels_[2],
                                                                           quant_level);
    }
    else {
        LYRA_CHECK_WITH_INFO(false, "controlnet_mode_ has to be either small or large");
    }

    controlnet_conditioning_embedding_weight =
        new ControlNetConditioningEmbeddingWeight<T>(controlnet_condition_channels_, block_out_channels_[0]);

    controlnet_final_conv_weight =
        new ControlNetFinalConvWeight<T>({320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280});
}

template<typename T>
XLControlNetModelWeight<T>::~XLControlNetModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);

        if (use_runtime_augemb_) {
            delete texttime_embedding_weight;
        }
        else {
            delete timestep_embedding_weight;
        }
        delete down_block_2d_weight;

        if (controlnet_mode_ == "large") {
            delete cross_attn_down_block_2d_weight1;
            delete cross_attn_down_block_2d_weight2;
            delete mid_block_2d_weight;
        }

        else if (controlnet_mode_ == "small") {
            delete small_down_block_2d_weight1;
            delete small_down_block_2d_weight2;
            delete small_mid_block_2d_weight;
        }
        else {
            LYRA_CHECK_WITH_INFO(false, "controlnet_mode_ has to be either small or large");
        }

        delete controlnet_conditioning_embedding_weight;
        delete controlnet_final_conv_weight;

        conv_in_weight = nullptr;
        conv_in_bias   = nullptr;
    }
}

template<typename T>
void XLControlNetModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&conv_in_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&conv_in_bias, block_out_channels_[0]);

        if (use_runtime_augemb_) {
            texttime_embedding_weight->mallocWeights();
        }
        else {
            timestep_embedding_weight->mallocWeights();
        }
        down_block_2d_weight->mallocWeights();

        if (controlnet_mode_ == "large") {
            cross_attn_down_block_2d_weight1->mallocWeights();
            cross_attn_down_block_2d_weight2->mallocWeights();
            mid_block_2d_weight->mallocWeights();
        }

        else if (controlnet_mode_ == "small") {
            small_down_block_2d_weight1->mallocWeights();
            small_down_block_2d_weight2->mallocWeights();
            small_mid_block_2d_weight->mallocWeights();
        }
        else {
            LYRA_CHECK_WITH_INFO(false, "controlnet_mode_ has to be either small or large");
        }

        controlnet_conditioning_embedding_weight->mallocWeights();
        controlnet_final_conv_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void XLControlNetModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
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

    if (use_runtime_augemb_) {
        texttime_embedding_weight->loadWeights(prefix, model_file_type);
    }
    else {
        timestep_embedding_weight->loadWeights(prefix + "time_embedding.", model_file_type);
    }

    down_block_2d_weight->loadWeights(prefix + "down_blocks.0.", model_file_type);

    if (controlnet_mode_ == "large") {
        cross_attn_down_block_2d_weight1->loadWeights(prefix + "down_blocks.1.", model_file_type);
        cross_attn_down_block_2d_weight2->loadWeights(prefix + "down_blocks.2.", model_file_type);
        mid_block_2d_weight->loadWeights(prefix + "mid_block.", model_file_type);
    }

    else if (controlnet_mode_ == "small") {
        small_down_block_2d_weight1->loadWeights(prefix + "down_blocks.1.", model_file_type);
        small_down_block_2d_weight2->loadWeights(prefix + "down_blocks.2.", model_file_type);
        small_mid_block_2d_weight->loadWeights(prefix + "mid_block.", model_file_type);
    }
    else {
        LYRA_CHECK_WITH_INFO(false, "controlnet_mode_ has to be either small or large");
    }

    controlnet_conditioning_embedding_weight->loadWeights(prefix + "controlnet_cond_embedding.", model_file_type);
    controlnet_final_conv_weight->loadWeights(prefix, model_file_type);
}

template<typename T>
void XLControlNetModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                      std::unordered_map<std::string, void*>& weights,
                                                      cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    controlnet_conditioning_embedding_weight->loadWeightsFromCache(
        prefix + "controlnet_cond_embedding.", weights, memcpyKind);
    controlnet_final_conv_weight->loadWeightsFromCache(prefix, weights, memcpyKind);

    void* tmp_conv_in_weight = weights[prefix + "conv_in.weight"];
    void* tmp_conv_in_bias   = weights[prefix + "conv_in.bias"];

    cudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);
    if (use_runtime_augemb_) {
        texttime_embedding_weight->loadWeightsFromCache(prefix, weights, memcpyKind);
    }
    else {
        timestep_embedding_weight->loadWeightsFromCache(prefix + "time_embedding.", weights, memcpyKind);
    }

    down_block_2d_weight->loadWeightsFromCache(prefix + "down_blocks.0.", weights, memcpyKind);

    if (controlnet_mode_ == "large") {
        cross_attn_down_block_2d_weight1->loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
        cross_attn_down_block_2d_weight2->loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);
        mid_block_2d_weight->loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);
    }

    else if (controlnet_mode_ == "small") {
        small_down_block_2d_weight1->loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
        small_down_block_2d_weight2->loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);
        small_mid_block_2d_weight->loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);
    }
    else {
        LYRA_CHECK_WITH_INFO(false, "controlnet_mode_ has to be either small or large");
    }

    // cout << "all weights loaded" << endl;
}

template<typename T>
std::string XLControlNetModelWeight<T>::getControlnetMode() const
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    return controlnet_mode_;

    // cout << "all weights loaded" << endl;
}

template class XLControlNetModelWeight<float>;
template class XLControlNetModelWeight<half>;
}  // namespace lyradiff