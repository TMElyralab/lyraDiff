#include "FluxTransformer2DModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxTransformer2DModelWeight<T>::FluxTransformer2DModelWeight(const size_t        input_channels,
                                                              const size_t        num_layers,
                                                              const size_t        num_single_layers,
                                                              const size_t        attention_head_dim,
                                                              const size_t        num_attention_heads,
                                                              const size_t        pooled_projection_dim,
                                                              const size_t        joint_attention_dim,
                                                              const bool          guidance_embeds,
                                                              const LyraQuantType quant_level,
                                                              IAllocator*         allocator):
    input_channels_(input_channels),
    num_layers_(num_layers),
    num_single_layers_(num_single_layers),
    attention_head_dim_(attention_head_dim),
    num_attention_heads_(num_attention_heads),
    pooled_projection_dim_(pooled_projection_dim),
    joint_attention_dim_(joint_attention_dim),
    guidance_embeds_(guidance_embeds)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    embedding_dim_     = attention_head_dim_ * num_attention_heads_;
    this->quant_level_ = quant_level;
    this->allocator_   = allocator;

    timestep_embedding_weight = new CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>(
        pooled_projection_dim_, embedding_dim_, 256, quant_level, allocator);

    if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL || this->quant_level_ == LyraQuantType::FP8_W8A8) {
        for (int i = 0; i < num_single_layers_; i++) {
            FluxSingleTransformerFP8BlockWeight<T>* tmp = new FluxSingleTransformerFP8BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 4, quant_level, allocator);
            single_transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "single_transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }

        for (int i = 0; i < num_layers_; i++) {
            FluxTransformerFP8BlockWeight<T>* tmp = new FluxTransformerFP8BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 6, quant_level, allocator);
            transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }
    }
    else if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL || this->quant_level_ == LyraQuantType::INT4_W4A4) {
        for (int i = 0; i < num_single_layers_; i++) {
            FluxSingleTransformerInt4BlockWeight<T>* tmp = new FluxSingleTransformerInt4BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 4, quant_level, allocator);
            single_transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "single_transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }

        for (int i = 0; i < num_layers_; i++) {
            FluxTransformerInt4BlockWeight<T>* tmp = new FluxTransformerInt4BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 6, quant_level, allocator);
            transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }
    }
    else {
        for (int i = 0; i < num_single_layers_; i++) {
            FluxSingleTransformerBlockWeight<T>* tmp = new FluxSingleTransformerBlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 4, quant_level, allocator);
            single_transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "single_transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }

        for (int i = 0; i < num_layers_; i++) {
            FluxTransformerBlockWeight<T>* tmp = new FluxTransformerBlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 6, quant_level, allocator);
            transformer_block_weight.push_back(tmp);

            std::string cur_layer_name           = "transformer_blocks_" + std::to_string(i);
            this->lora_layer_map[cur_layer_name] = tmp;
        }
    }

    norm_weight = new AdaLayerNormWeight<T>(embedding_dim_, 2, quant_level, allocator);

    this->lora_layer_map["time_text_embed"] = timestep_embedding_weight;
    this->lora_layer_map["norm_out"]        = norm_weight;
}

template<typename T>
FluxTransformer2DModelWeight<T>::~FluxTransformer2DModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(x_embedder_weight);
        deviceFree(x_embedder_bias);
        deviceFree(context_embedder_weight);
        deviceFree(context_embedder_bias);
        deviceFree(proj_out_weight);
        deviceFree(proj_out_bias);

        x_embedder_weight = nullptr;
        x_embedder_bias   = nullptr;

        context_embedder_weight = nullptr;
        context_embedder_bias   = nullptr;

        proj_out_weight = nullptr;
        proj_out_bias   = nullptr;

        delete timestep_embedding_weight;
        delete norm_weight;

        for (int i = 0; i < num_single_layers_; i++) {
            delete single_transformer_block_weight[i];
        }

        for (int i = 0; i < num_layers_; i++) {
            delete transformer_block_weight[i];
        }

        if (this->quant_level_ != LyraQuantType::NONE) {
            free(x_embedder_weight_h);
            free(context_embedder_weight_h);
            free(proj_out_weight_h);
        }
    }
}

template<typename T>
void FluxTransformer2DModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&x_embedder_weight, embedding_dim_ * input_channels_);
        deviceMalloc(&x_embedder_bias, embedding_dim_);
        deviceMalloc(&context_embedder_weight, embedding_dim_ * joint_attention_dim_);
        deviceMalloc(&context_embedder_bias, embedding_dim_);
        deviceMalloc(&proj_out_weight, embedding_dim_ * input_channels_);
        deviceMalloc(&proj_out_bias, input_channels_);

        timestep_embedding_weight->mallocWeights();
        norm_weight->mallocWeights();
        for (int i = 0; i < num_single_layers_; i++) {
            single_transformer_block_weight[i]->mallocWeights();
        }

        for (int i = 0; i < num_layers_; i++) {
            transformer_block_weight[i]->mallocWeights();
        }

        if (this->quant_level_ != LyraQuantType::NONE) {
            x_embedder_weight_h       = (T*)malloc(sizeof(T) * embedding_dim_ * input_channels_);
            context_embedder_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * joint_attention_dim_);
            proj_out_weight_h         = (T*)malloc(sizeof(T) * embedding_dim_ * input_channels_);

            this->lora_weight_map = {
                {"x_embedder",
                 new LoraWeightV2<T>({embedding_dim_, input_channels_}, x_embedder_weight, x_embedder_weight_h)},
                {"context_embedder",
                 new LoraWeightV2<T>(
                     {embedding_dim_, joint_attention_dim_}, context_embedder_weight, context_embedder_weight_h)},
                {"proj_out",
                 new LoraWeightV2<T>({embedding_dim_, input_channels_}, proj_out_weight, proj_out_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"x_embedder", new LoraWeight<T>({embedding_dim_, input_channels_}, x_embedder_weight)},
                {"context_embedder",
                 new LoraWeight<T>({embedding_dim_, joint_attention_dim_}, context_embedder_weight)},
                {"proj_out", new LoraWeight<T>({embedding_dim_, input_channels_}, proj_out_weight)}};
        }
    }

    is_maintain_buffer = true;
}

// template<typename T>
// void Unet2dConditionalModelWeight<T>::loadLoraFromWeight(std::string                          lora_path,
//                                                          std::string                          prefix,
//                                                          std::unordered_map<std::string, T*>& lora_weights,
//                                                          float                                lora_alpha,
//                                                          FtCudaDataType                       lora_file_type,
//                                                          cudaStream_t                         stream)
// {
//     FT_LOG_DEBUG(__PRETTY_FUNCTION__);
//     cross_attn_down_block_2d_weight1->loadLoraFromWeight(
//         lora_path, prefix + "down_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);
//     cross_attn_down_block_2d_weight2->loadLoraFromWeight(
//         lora_path, prefix + "down_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);
//     cross_attn_down_block_2d_weight3->loadLoraFromWeight(
//         lora_path, prefix + "down_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);
//     down_block_2d_weight->loadLoraFromWeight(
//         lora_path, prefix + "down_blocks_3_", lora_weights, lora_alpha, lora_file_type, stream);
//     mid_block_2d_weight->loadLoraFromWeight(
//         lora_path, prefix + "mid_block_", lora_weights, lora_alpha, lora_file_type, stream);
//     up_block_2d_weight->loadLoraFromWeight(
//         lora_path, prefix + "up_blocks_0_", lora_weights, lora_alpha, lora_file_type, stream);
//     cross_attn_up_block_2d_weight1->loadLoraFromWeight(
//         lora_path, prefix + "up_blocks_1_", lora_weights, lora_alpha, lora_file_type, stream);
//     cross_attn_up_block_2d_weight2->loadLoraFromWeight(
//         lora_path, prefix + "up_blocks_2_", lora_weights, lora_alpha, lora_file_type, stream);
//     cross_attn_up_block_2d_weight3->loadLoraFromWeight(
//         lora_path, prefix + "up_blocks_3_", lora_weights, lora_alpha, lora_file_type, stream);
// }

// template<typename T>
// void Unet2dConditionalModelWeight<T>::loadLoraFromCache(std::string                          prefix,
//                                                         std::unordered_map<std::string, T*>& lora_weights,
//                                                         float                                lora_alpha,
//                                                         bool                                 from_outside)
// {
//     FT_LOG_DEBUG(__PRETTY_FUNCTION__);
//     cross_attn_down_block_2d_weight1->loadLoraFromCache(
//         prefix + "down_blocks_0_", lora_weights, lora_alpha, from_outside);
//     cross_attn_down_block_2d_weight2->loadLoraFromCache(
//         prefix + "down_blocks_1_", lora_weights, lora_alpha, from_outside);
//     cross_attn_down_block_2d_weight3->loadLoraFromCache(
//         prefix + "down_blocks_2_", lora_weights, lora_alpha, from_outside);
//     down_block_2d_weight->loadLoraFromCache(prefix + "down_blocks_3_", lora_weights, lora_alpha, from_outside);
//     mid_block_2d_weight->loadLoraFromCache(prefix + "mid_block_", lora_weights, lora_alpha, from_outside);
//     up_block_2d_weight->loadLoraFromCache(prefix + "up_blocks_0_", lora_weights, lora_alpha, from_outside);
//     cross_attn_up_block_2d_weight1->loadLoraFromCache(prefix + "up_blocks_1_", lora_weights, lora_alpha,
//     from_outside); cross_attn_up_block_2d_weight2->loadLoraFromCache(prefix + "up_blocks_2_", lora_weights,
//     lora_alpha, from_outside); cross_attn_up_block_2d_weight3->loadLoraFromCache(prefix + "up_blocks_3_",
//     lora_weights, lora_alpha, from_outside);
// }

template<typename T>
void FluxTransformer2DModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    loadWeightFromBin<T>(
        x_embedder_weight, {embedding_dim_ * input_channels_}, prefix + "x_embedder.weight.bin", model_file_type);
    loadWeightFromBin<T>(x_embedder_bias, {embedding_dim_}, prefix + "x_embedder.bias.bin", model_file_type);
    loadWeightFromBin<T>(context_embedder_weight,
                         {embedding_dim_ * joint_attention_dim_},
                         prefix + "context_embedder.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        context_embedder_bias, {embedding_dim_}, prefix + "context_embedder.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        proj_out_weight, {embedding_dim_ * input_channels_}, prefix + "proj_out.weight.bin", model_file_type);
    loadWeightFromBin<T>(proj_out_bias, {input_channels_}, prefix + "proj_out.bias.bin", model_file_type);

    timestep_embedding_weight->loadWeights(prefix + "time_text_embed.", model_file_type);
    norm_weight->loadWeights(prefix + "norm_out.", model_file_type);

    for (int i = 0; i < num_single_layers_; i++) {
        single_transformer_block_weight[i]->loadWeights(prefix + "single_transformer_blocks." + std::to_string(i) + ".",
                                                        model_file_type);
        cout << "loaded single_transformer_block_weight: " << i << endl;
    }

    for (int i = 0; i < num_layers_; i++) {
        transformer_block_weight[i]->loadWeights(prefix + "transformer_blocks." + std::to_string(i) + ".",
                                                 model_file_type);

        cout << "loaded transformer_block_weight: " << i << endl;
    }
    if (this->quant_level_ != LyraQuantType::NONE) {
        cout << "FluxTransformer2DModelWeight load weights cur quant level: " << this->quant_level_ << endl;

        cudaMemcpyAsync(x_embedder_weight_h,
                        x_embedder_weight,
                        sizeof(T) * embedding_dim_ * input_channels_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(context_embedder_weight_h,
                        context_embedder_weight,
                        sizeof(T) * embedding_dim_ * joint_attention_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(proj_out_weight_h,
                        proj_out_weight,
                        sizeof(T) * embedding_dim_ * input_channels_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template<typename T>
void FluxTransformer2DModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                           std::unordered_map<std::string, void*>& weights,
                                                           cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_x_embedder_weight       = weights[prefix + "x_embedder.weight"];
    void* tmp_x_embedder_bias         = weights[prefix + "x_embedder.bias"];
    void* tmp_context_embedder_weight = weights[prefix + "context_embedder.weight"];
    void* tmp_context_embedder_bias   = weights[prefix + "context_embedder.bias"];
    void* tmp_proj_out_weight         = weights[prefix + "proj_out.weight"];
    void* tmp_proj_out_bias           = weights[prefix + "proj_out.bias"];

    cudaMemcpy(x_embedder_weight, tmp_x_embedder_weight, sizeof(T) * embedding_dim_ * input_channels_, memcpyKind);
    cudaMemcpy(x_embedder_bias, tmp_x_embedder_bias, sizeof(T) * embedding_dim_, memcpyKind);
    cudaMemcpy(context_embedder_weight,
               tmp_context_embedder_weight,
               sizeof(T) * embedding_dim_ * joint_attention_dim_,
               memcpyKind);
    cudaMemcpy(context_embedder_bias, tmp_context_embedder_bias, sizeof(T) * embedding_dim_, memcpyKind);
    cudaMemcpy(proj_out_weight, tmp_proj_out_weight, sizeof(T) * embedding_dim_ * input_channels_, memcpyKind);
    cudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * input_channels_, memcpyKind);

    timestep_embedding_weight->loadWeightsFromCache(prefix + "time_text_embed.", weights, memcpyKind);
    norm_weight->loadWeightsFromCache(prefix + "norm_out.", weights, memcpyKind);

    for (int i = 0; i < num_single_layers_; i++) {
        single_transformer_block_weight[i]->loadWeightsFromCache(
            prefix + "single_transformer_blocks." + std::to_string(i) + ".", weights, memcpyKind);
        cout << "loaded single_transformer_block_weight: " << i << endl;
    }

    check_cuda_error(cudaGetLastError());

    for (int i = 0; i < num_layers_; i++) {
        transformer_block_weight[i]->loadWeightsFromCache(
            prefix + "transformer_blocks." + std::to_string(i) + ".", weights, memcpyKind);
    }

    if (this->quant_level_ != LyraQuantType::NONE) {
        cout << "FluxTransformer2DModelWeight load weights cur quant level: " << this->quant_level_ << endl;

        cudaMemcpyAsync(x_embedder_weight_h,
                        x_embedder_weight,
                        sizeof(T) * embedding_dim_ * input_channels_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(context_embedder_weight_h,
                        context_embedder_weight,
                        sizeof(T) * embedding_dim_ * joint_attention_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(proj_out_weight_h,
                        proj_out_weight,
                        sizeof(T) * embedding_dim_ * input_channels_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template class FluxTransformer2DModelWeight<float>;
template class FluxTransformer2DModelWeight<half>;
#ifdef ENABLE_BF16
template class FluxTransformer2DModelWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff