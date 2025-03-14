#include "GLVUnet2dConditionalModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
GLVUnet2dConditionalModelWeight<T>::GLVUnet2dConditionalModelWeight(const size_t        input_channels,
                                                                    const size_t        output_channels,
                                                                    const bool          use_runtime_augemb,
                                                                    const LyraQuantType quant_level):
    use_runtime_augemb_(use_runtime_augemb), input_channels_(input_channels), output_channels_(output_channels)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (use_runtime_augemb_) {
        texttime_embedding_weight = new TextTimeEmbeddingBlockWeight<T>(
            block_out_channels_[0], add_emb_input_dim_, temb_channels_, temb_channels_, quant_level);
    }
    else {
        timestep_embedding_weight =
            new TimestepEmbeddingBlockWeight<T>(block_out_channels_[0], temb_channels_, temb_channels_, quant_level);
    }

    down_block_2d_weight =
        new XLDownBlock2DWeight<T>(block_out_channels_[0], block_out_channels_[0], block_out_channels_[0], true);

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

    cross_attn_up_block_2d_weight1 = new GLVCrossAttnUpBlock2dWeight<T>(block_out_channels_[1],
                                                                        block_out_channels_[2],
                                                                        block_out_channels_[2],
                                                                        temb_channels_,
                                                                        head_nums_[1],
                                                                        cross_attn_dim_,
                                                                        norm_num_groups_,
                                                                        inner_trans_nums_[1],
                                                                        quant_level);

    cross_attn_up_block_2d_weight2 = new GLVCrossAttnUpBlock2dWeight<T>(block_out_channels_[0],
                                                                        block_out_channels_[1],
                                                                        block_out_channels_[2],
                                                                        temb_channels_,
                                                                        head_nums_[0],
                                                                        cross_attn_dim_,
                                                                        norm_num_groups_,
                                                                        inner_trans_nums_[0],
                                                                        quant_level);

    up_block_2d_weight = new GLVUpBlock2dWeight<T>(
        block_out_channels_[0], block_out_channels_[0], block_out_channels_[1], norm_num_groups_);

    mid_block_project_module_weight = new ZeroSFTWeight<T>(block_out_channels_[2], block_out_channels_[2], 0);

    this->vec_basic_transformer_container_weights = {cross_attn_down_block_2d_weight1,
                                                     cross_attn_down_block_2d_weight2,
                                                     cross_attn_up_block_2d_weight1,
                                                     cross_attn_up_block_2d_weight2,
                                                     mid_block_2d_weight};

    this->lora_layer_map = {{"down_blocks_0", down_block_2d_weight},
                            {"down_blocks_1", cross_attn_down_block_2d_weight1},
                            {"down_blocks_2", cross_attn_down_block_2d_weight2},
                            {"mid_block", mid_block_2d_weight},
                            {"up_blocks_0", cross_attn_up_block_2d_weight1},
                            {"up_blocks_1", cross_attn_up_block_2d_weight2},
                            {"up_blocks_2", up_block_2d_weight}};
}

template<typename T>
GLVUnet2dConditionalModelWeight<T>::~GLVUnet2dConditionalModelWeight()
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

        if (use_runtime_augemb_) {
            delete texttime_embedding_weight;
        }
        else {
            delete timestep_embedding_weight;
        }
        delete down_block_2d_weight;
        delete cross_attn_down_block_2d_weight1;
        delete cross_attn_down_block_2d_weight2;
        delete mid_block_2d_weight;
        delete cross_attn_up_block_2d_weight1;
        delete cross_attn_up_block_2d_weight2;
        delete up_block_2d_weight;

        texttime_embedding_weight        = nullptr;
        timestep_embedding_weight        = nullptr;
        down_block_2d_weight             = nullptr;
        cross_attn_down_block_2d_weight1 = nullptr;
        cross_attn_down_block_2d_weight2 = nullptr;
        mid_block_2d_weight              = nullptr;
        cross_attn_up_block_2d_weight1   = nullptr;
        cross_attn_up_block_2d_weight2   = nullptr;
        up_block_2d_weight               = nullptr;
    }
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&conv_in_weight, input_channels_ * 3 * 3 * block_out_channels_[0]);
        deviceMalloc(&conv_in_bias, block_out_channels_[0]);
        deviceMalloc(&conv_out_norm_beta, block_out_channels_[0]);
        deviceMalloc(&conv_out_norm_gamma, block_out_channels_[0]);
        deviceMalloc(&conv_out_weight, block_out_channels_[0] * 3 * 3 * output_channels_);
        deviceMalloc(&conv_out_bias, output_channels_);

        if (use_runtime_augemb_) {
            texttime_embedding_weight->mallocWeights();
        }
        else {
            timestep_embedding_weight->mallocWeights();
        }
        down_block_2d_weight->mallocWeights();
        cross_attn_down_block_2d_weight1->mallocWeights();
        cross_attn_down_block_2d_weight2->mallocWeights();

        mid_block_2d_weight->mallocWeights();
        cross_attn_up_block_2d_weight1->mallocWeights();
        cross_attn_up_block_2d_weight2->mallocWeights();
        up_block_2d_weight->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::mallocProjectModuleWeights()
{
    if (!is_maintain_project_buffer) {
        mid_block_project_module_weight->mallocWeights();
        cross_attn_up_block_2d_weight1->mallocProjectModuleWeights();
        cross_attn_up_block_2d_weight2->mallocProjectModuleWeights();
        up_block_2d_weight->mallocProjectModuleWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                            std::string                          prefix,
                                                            std::unordered_map<std::string, T*>& lora_weights,
                                                            float                                lora_alpha,
                                                            FtCudaDataType                       lora_file_type,
                                                            cudaStream_t                         stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    down_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);

    cross_attn_down_block_2d_weight1->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);

    cross_attn_down_block_2d_weight2->loadLoraFromWeight(
        lora_path, prefix + "down_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);

    mid_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "mid_block_", lora_weights, lora_alpha, lora_file_type, stream);

    cross_attn_up_block_2d_weight1->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);

    cross_attn_up_block_2d_weight2->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);

    up_block_2d_weight->loadLoraFromWeight(
        lora_path, prefix + "up_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadLoraFromCache(std::string                          prefix,
                                                           std::unordered_map<std::string, T*>& lora_weights,
                                                           float                                lora_alpha,
                                                           bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    down_block_2d_weight->loadLoraFromCache(prefix + "down_blocks_0_", lora_weights, lora_alpha, from_outside);
    cross_attn_down_block_2d_weight1->loadLoraFromCache(
        prefix + "down_blocks_1_", lora_weights, lora_alpha, from_outside);
    cross_attn_down_block_2d_weight2->loadLoraFromCache(
        prefix + "down_blocks_2_", lora_weights, lora_alpha, from_outside);

    mid_block_2d_weight->loadLoraFromCache(prefix + "mid_block_", lora_weights, lora_alpha, from_outside);

    cross_attn_up_block_2d_weight1->loadLoraFromCache(prefix + "up_blocks_0_", lora_weights, lora_alpha, from_outside);
    cross_attn_up_block_2d_weight2->loadLoraFromCache(prefix + "up_blocks_1_", lora_weights, lora_alpha, from_outside);
    up_block_2d_weight->loadLoraFromCache(prefix + "up_blocks_2_", lora_weights, lora_alpha, from_outside);
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
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
    if (use_runtime_augemb_) {
        texttime_embedding_weight->loadWeights(prefix, model_file_type);
    }
    else {
        timestep_embedding_weight->loadWeights(prefix + "time_embedding.", model_file_type);
    }

    down_block_2d_weight->loadWeights(prefix + "down_blocks.0.", model_file_type);
    cross_attn_down_block_2d_weight1->loadWeights(prefix + "down_blocks.1.", model_file_type);
    cross_attn_down_block_2d_weight2->loadWeights(prefix + "down_blocks.2.", model_file_type);

    mid_block_2d_weight->loadWeights(prefix + "mid_block.", model_file_type);

    cross_attn_up_block_2d_weight1->loadWeights(prefix + "up_blocks.0.", model_file_type);
    cross_attn_up_block_2d_weight2->loadWeights(prefix + "up_blocks.1.", model_file_type);
    up_block_2d_weight->loadWeights(prefix + "up_blocks.2.", model_file_type);
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadProjectWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocProjectModuleWeights();
    }

    mid_block_project_module_weight->loadWeights(prefix + "mid_block.project_modules.", model_file_type);
    cross_attn_up_block_2d_weight1->loadProjectWeights(prefix + "up_blocks.0.", model_file_type);
    cross_attn_up_block_2d_weight2->loadProjectWeights(prefix + "up_blocks.1.", model_file_type);
    up_block_2d_weight->loadProjectWeights(prefix + "up_blocks.2.", model_file_type);
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadProjectWeightsFromCache(std::string                             prefix,
                                                                     std::unordered_map<std::string, void*>& weights,
                                                                     cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocProjectModuleWeights();
    }

    mid_block_project_module_weight->loadWeightsFromCache(prefix + "mid_block.project_modules.", weights, memcpyKind);
    cross_attn_up_block_2d_weight1->loadProjectWeightsFromCache(prefix + "up_blocks.0.", weights, memcpyKind);
    cross_attn_up_block_2d_weight2->loadProjectWeightsFromCache(prefix + "up_blocks.1.", weights, memcpyKind);
    up_block_2d_weight->loadProjectWeightsFromCache(prefix + "up_blocks.2.", weights, memcpyKind);
}

template<typename T>
void GLVUnet2dConditionalModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
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

    cudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * input_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_out_norm_gamma, tmp_conv_out_norm_gamma, sizeof(T) * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_out_norm_beta, tmp_conv_out_norm_beta, sizeof(T) * block_out_channels_[0], memcpyKind);
    cudaMemcpy(conv_out_weight,
               tmp_conv_out_weight,
               sizeof(T) * block_out_channels_[0] * 3 * 3 * output_channels_,
               memcpyKind);
    cudaMemcpy(conv_out_bias, tmp_conv_out_bias, sizeof(T) * output_channels_, memcpyKind);
    if (use_runtime_augemb_) {
        texttime_embedding_weight->loadWeightsFromCache(prefix, weights, memcpyKind);
    }
    else {
        timestep_embedding_weight->loadWeightsFromCache(prefix + "time_embedding.", weights, memcpyKind);
    }

    down_block_2d_weight->loadWeightsFromCache(prefix + "down_blocks.0.", weights, memcpyKind);
    cross_attn_down_block_2d_weight1->loadWeightsFromCache(prefix + "down_blocks.1.", weights, memcpyKind);
    cross_attn_down_block_2d_weight2->loadWeightsFromCache(prefix + "down_blocks.2.", weights, memcpyKind);

    mid_block_2d_weight->loadWeightsFromCache(prefix + "mid_block.", weights, memcpyKind);

    cross_attn_up_block_2d_weight1->loadWeightsFromCache(prefix + "up_blocks.0.", weights, memcpyKind);
    cross_attn_up_block_2d_weight2->loadWeightsFromCache(prefix + "up_blocks.1.", weights, memcpyKind);
    up_block_2d_weight->loadWeightsFromCache(prefix + "up_blocks.2.", weights, memcpyKind);
}

template class GLVUnet2dConditionalModelWeight<float>;
template class GLVUnet2dConditionalModelWeight<half>;
}  // namespace lyradiff