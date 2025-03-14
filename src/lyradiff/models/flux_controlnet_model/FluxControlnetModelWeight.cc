#include "FluxControlnetModelWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxControlnetModelWeight<T>::FluxControlnetModelWeight(const size_t        input_channels,
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

    if (this->quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < num_single_layers_; i++) {
            FluxSingleTransformerFP8BlockWeight<T>* tmp = new FluxSingleTransformerFP8BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 4, quant_level, allocator);
            single_transformer_block_weight.push_back(tmp);
        }

        for (int i = 0; i < num_layers_; i++) {
            FluxTransformerFP8BlockWeight<T>* tmp = new FluxTransformerFP8BlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 6, quant_level, allocator);
            transformer_block_weight.push_back(tmp);
        }
    }
    else {
        for (int i = 0; i < num_single_layers_; i++) {
            FluxSingleTransformerBlockWeight<T>* tmp = new FluxSingleTransformerBlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 4, quant_level, allocator);
            single_transformer_block_weight.push_back(tmp);
        }

        for (int i = 0; i < num_layers_; i++) {
            FluxTransformerBlockWeight<T>* tmp = new FluxTransformerBlockWeight<T>(
                embedding_dim_, num_attention_heads_, attention_head_dim_, 6, quant_level, allocator);
            transformer_block_weight.push_back(tmp);
        }
    }
}

template<typename T>
FluxControlnetModelWeight<T>::~FluxControlnetModelWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(x_embedder_weight);
        deviceFree(x_embedder_bias);
        deviceFree(context_embedder_weight);
        deviceFree(context_embedder_bias);
        deviceFree(controlnet_x_embedder_weight);
        deviceFree(controlnet_x_embedder_bias);
        // deviceFree(proj_out_weight);
        // deviceFree(proj_out_bias);

        x_embedder_weight = nullptr;
        x_embedder_bias   = nullptr;

        context_embedder_weight = nullptr;
        context_embedder_bias   = nullptr;

        // proj_out_weight = nullptr;
        // proj_out_bias   = nullptr;

        delete timestep_embedding_weight;

        for (int i = 0; i < num_single_layers_; i++) {
            delete single_transformer_block_weight[i];
        }

        for (int i = 0; i < num_layers_; i++) {
            delete transformer_block_weight[i];
        }
    }
}

template<typename T>
void FluxControlnetModelWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&x_embedder_weight, embedding_dim_ * input_channels_);
        deviceMalloc(&x_embedder_bias, embedding_dim_);
        deviceMalloc(&controlnet_x_embedder_weight, embedding_dim_ * input_channels_);
        deviceMalloc(&controlnet_x_embedder_bias, embedding_dim_);
        deviceMalloc(&context_embedder_weight, embedding_dim_ * joint_attention_dim_);
        deviceMalloc(&context_embedder_bias, embedding_dim_);
        // deviceMalloc(&proj_out_weight, embedding_dim_ * input_channels_);
        // deviceMalloc(&proj_out_bias, input_channels_);

        timestep_embedding_weight->mallocWeights();

        for (int i = 0; i < num_single_layers_; i++) {
            single_transformer_block_weight[i]->mallocWeights();
        }

        for (int i = 0; i < num_layers_; i++) {
            transformer_block_weight[i]->mallocWeights();
            T* cur_weight;
            T* cur_bias;
            deviceMalloc(&cur_weight, embedding_dim_ * embedding_dim_);
            deviceMalloc(&cur_bias, embedding_dim_);
            control_weights.push_back(cur_weight);
            control_bias.push_back(cur_bias);
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxControlnetModelWeight<T>::updateConfig(const size_t num_layers, const size_t num_single_layers)
{
    // 如果新的config中num layers 比当前的大，需要创建新的weight层
    if (num_layers > num_layers_) {
        for (int i = 0; i < num_layers - num_layers_; i++) {
            if (this->quant_level_ != LyraQuantType::NONE) {
                FluxTransformerFP8BlockWeight<T>* tmp = new FluxTransformerFP8BlockWeight<T>(
                    embedding_dim_, num_attention_heads_, attention_head_dim_, 6, this->quant_level_, this->allocator_);
                tmp->mallocWeights();
                transformer_block_weight.push_back(tmp);
            }
            else {
                FluxTransformerBlockWeight<T>* tmp = new FluxTransformerBlockWeight<T>(
                    embedding_dim_, num_attention_heads_, attention_head_dim_, 4, this->quant_level_, this->allocator_);
                tmp->mallocWeights();
                transformer_block_weight.push_back(tmp);
            }

            T* cur_weight;
            T* cur_bias;
            deviceMalloc(&cur_weight, embedding_dim_ * embedding_dim_);
            deviceMalloc(&cur_bias, embedding_dim_);
            control_weights.push_back(cur_weight);
            control_bias.push_back(cur_bias);
        }

        num_layers_ = num_layers;
    }
    else if (num_layers < num_layers_) {
        for (int i = num_layers; i < num_layers_; i++) {
            delete transformer_block_weight[i];
            deviceFree(control_weights[i]);
            deviceFree(control_bias[i]);
        }

        for (int i = 0; i < num_layers_ - num_layers; i++) {
            transformer_block_weight.pop_back();
            control_weights.pop_back();
            control_bias.pop_back();
        }

        num_layers_ = num_layers;
    }
}

template<typename T>
size_t FluxControlnetModelWeight<T>::getNumLayers()
{
    return num_layers_;
}

template<typename T>
size_t FluxControlnetModelWeight<T>::getNumSingleLayers()
{
    return num_single_layers_;
}

template<typename T>
void FluxControlnetModelWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    loadWeightFromBin<T>(
        x_embedder_weight, {embedding_dim_ * input_channels_}, prefix + "x_embedder.weight.bin", model_file_type);
    loadWeightFromBin<T>(x_embedder_bias, {embedding_dim_}, prefix + "x_embedder.bias.bin", model_file_type);

    loadWeightFromBin<T>(controlnet_x_embedder_weight,
                         {embedding_dim_ * input_channels_},
                         prefix + "controlnet_x_embedder.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        controlnet_x_embedder_bias, {embedding_dim_}, prefix + "controlnet_x_embedder.bias.bin", model_file_type);

    loadWeightFromBin<T>(context_embedder_weight,
                         {embedding_dim_ * joint_attention_dim_},
                         prefix + "context_embedder.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        context_embedder_bias, {embedding_dim_}, prefix + "context_embedder.bias.bin", model_file_type);

    // loadWeightFromBin<T>(
    //     proj_out_weight, {embedding_dim_ * input_channels_}, prefix + "proj_out.weight.bin", model_file_type);
    // loadWeightFromBin<T>(proj_out_bias, {input_channels_}, prefix + "proj_out.bias.bin", model_file_type);

    timestep_embedding_weight->loadWeights(prefix + "time_text_embed.", model_file_type);

    for (int i = 0; i < num_single_layers_; i++) {
        single_transformer_block_weight[i]->loadWeights(prefix + "single_transformer_blocks." + std::to_string(i) + ".",
                                                        model_file_type);
        cout << "loaded single_transformer_block_weight: " << i << endl;
    }

    for (int i = 0; i < num_layers_; i++) {
        transformer_block_weight[i]->loadWeights(prefix + "transformer_blocks." + std::to_string(i) + ".",
                                                 model_file_type);

        loadWeightFromBin<T>(control_weights[i],
                             {embedding_dim_ * embedding_dim_},
                             prefix + "controlnet_blocks." + std::to_string(i) + ".weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(control_bias[i],
                             {embedding_dim_},
                             prefix + "controlnet_blocks." + std::to_string(i) + ".bias.bin",
                             model_file_type);

        cout << "loaded transformer_block_weight: " << i << endl;
    }
}

template<typename T>
void FluxControlnetModelWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                        std::unordered_map<std::string, void*>& weights,
                                                        cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_x_embedder_weight            = weights[prefix + "x_embedder.weight"];
    void* tmp_x_embedder_bias              = weights[prefix + "x_embedder.bias"];
    void* tmp_controlnet_x_embedder_weight = weights[prefix + "controlnet_x_embedder.weight"];
    void* tmp_controlnet_x_embedder_bias   = weights[prefix + "controlnet_x_embedder.bias"];
    void* tmp_context_embedder_weight      = weights[prefix + "context_embedder.weight"];
    void* tmp_context_embedder_bias        = weights[prefix + "context_embedder.bias"];
    // void* tmp_proj_out_weight              = weights[prefix + "proj_out.weight"];
    // void* tmp_proj_out_bias                = weights[prefix + "proj_out.bias"];

    cudaMemcpy(x_embedder_weight, tmp_x_embedder_weight, sizeof(T) * embedding_dim_ * input_channels_, memcpyKind);
    cudaMemcpy(x_embedder_bias, tmp_x_embedder_bias, sizeof(T) * embedding_dim_, memcpyKind);
    cudaMemcpy(controlnet_x_embedder_weight,
               tmp_controlnet_x_embedder_weight,
               sizeof(T) * embedding_dim_ * input_channels_,
               memcpyKind);
    cudaMemcpy(controlnet_x_embedder_bias, tmp_controlnet_x_embedder_bias, sizeof(T) * embedding_dim_, memcpyKind);
    cudaMemcpy(context_embedder_weight,
               tmp_context_embedder_weight,
               sizeof(T) * embedding_dim_ * joint_attention_dim_,
               memcpyKind);
    cudaMemcpy(context_embedder_bias, tmp_context_embedder_bias, sizeof(T) * embedding_dim_, memcpyKind);
    // cudaMemcpy(proj_out_weight, tmp_proj_out_weight, sizeof(T) * embedding_dim_ * input_channels_, memcpyKind);
    // cudaMemcpy(proj_out_bias, tmp_proj_out_bias, sizeof(T) * input_channels_, memcpyKind);

    // check_cuda_error(cudaGetLastError());

    timestep_embedding_weight->loadWeightsFromCache(prefix + "time_text_embed.", weights, memcpyKind);

    check_cuda_error(cudaGetLastError());

    for (int i = 0; i < num_single_layers_; i++) {
        cout << "loaded single_transformer_block_weight: " << i << endl;

        single_transformer_block_weight[i]->loadWeightsFromCache(
            prefix + "single_transformer_blocks." + std::to_string(i) + ".", weights, memcpyKind);
    }

    for (int i = 0; i < num_layers_; i++) {
        transformer_block_weight[i]->loadWeightsFromCache(
            prefix + "transformer_blocks." + std::to_string(i) + ".", weights, memcpyKind);

        void* tmp_weight = weights[prefix + "controlnet_blocks." + std::to_string(i) + ".weight"];
        void* tmp_bias   = weights[prefix + "controlnet_blocks." + std::to_string(i) + ".bias"];
        check_cuda_error(
            cudaMemcpy(control_weights[i], tmp_weight, sizeof(T) * embedding_dim_ * embedding_dim_, memcpyKind));
        check_cuda_error(cudaMemcpy(control_bias[i], tmp_bias, sizeof(T) * embedding_dim_, memcpyKind));

        cout << "loaded transformer_block_weight: " << i << endl;
    }
}

template class FluxControlnetModelWeight<float>;
template class FluxControlnetModelWeight<half>;
#ifdef ENABLE_BF16
template class FluxControlnetModelWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff