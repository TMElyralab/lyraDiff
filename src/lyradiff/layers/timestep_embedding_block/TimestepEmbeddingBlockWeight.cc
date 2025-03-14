#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {
template<typename T>
TimestepEmbeddingBlockWeight<T>::TimestepEmbeddingBlockWeight(const size_t        input_dim_,
                                                              const size_t        output_dim_0_,
                                                              const size_t        output_dim_,
                                                              const LyraQuantType quant_level,
                                                              IAllocator*         allocator):
    input_dim_(input_dim_), output_dim_0_(output_dim_0_), output_dim_(output_dim_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->quant_level_ = quant_level;
    this->allocator_   = allocator;
}

template<typename T>
TimestepEmbeddingBlockWeight<T>::~TimestepEmbeddingBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(linear1_weight);
        deviceFree(linear1_bias);
        deviceFree(linear2_weight);
        deviceFree(linear2_bias);

        linear1_weight = nullptr;
        linear1_bias   = nullptr;

        linear2_weight = nullptr;
        linear2_bias   = nullptr;
    }
}

template<typename T>
void TimestepEmbeddingBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        deviceMalloc(&linear1_weight, input_dim_ * output_dim_0_);
        deviceMalloc(&linear1_bias, output_dim_0_);

        deviceMalloc(&linear2_weight, output_dim_0_ * output_dim_);
        deviceMalloc(&linear2_bias, output_dim_);

        // this->lora_weight_size_map = {{"linear_1", input_dim_ * output_dim_0_},
        //                               {"linear_2", output_dim_0_ * output_dim_}};
        // this->lora_weight_map      = {{"linear_1", linear1_weight}, {"linear_2", linear2_weight}};
        if (this->quant_level_ != LyraQuantType::NONE) {
            linear1_weight_h = (T*)malloc(sizeof(T) * input_dim_ * output_dim_0_);
            linear2_weight_h = (T*)malloc(sizeof(T) * output_dim_0_ * output_dim_);

            this->lora_weight_map = {
                {"linear_1", new LoraWeightV2<T>({input_dim_, output_dim_0_}, linear1_weight, linear1_weight_h)},
                {"linear_2", new LoraWeightV2<T>({output_dim_0_, output_dim_}, linear2_weight, linear2_weight_h)}};
        }
        else {
            this->lora_weight_map = {{"linear_1", new LoraWeight<T>({input_dim_, output_dim_0_}, linear1_weight)},
                                     {"linear_2", new LoraWeight<T>({output_dim_0_, output_dim_}, linear2_weight)}};
        }
    }
    is_maintain_buffer = true;
}

template<typename T>
void TimestepEmbeddingBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    loadWeightFromBin<T>(linear1_weight, {input_dim_, output_dim_0_}, prefix + "linear_1.weight.bin", model_file_type);
    loadWeightFromBin<T>(linear1_bias, {output_dim_0_}, prefix + "linear_1.bias.bin", model_file_type);

    loadWeightFromBin<T>(linear2_weight, {output_dim_0_, output_dim_}, prefix + "linear_2.weight.bin", model_file_type);
    loadWeightFromBin<T>(linear2_bias, {output_dim_}, prefix + "linear_2.bias.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(linear1_weight_h,
                        linear1_weight,
                        sizeof(T) * input_dim_ * output_dim_0_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(linear2_weight_h,
                        linear2_weight,
                        sizeof(T) * output_dim_0_ * output_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template<typename T>
void TimestepEmbeddingBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                           std::unordered_map<std::string, void*>& weights,
                                                           cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // linear weights
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_linear1_weight = weights[prefix + "linear_1.weight"];
    void* tmp_linear1_bias   = weights[prefix + "linear_1.bias"];
    void* tmp_linear2_weight = weights[prefix + "linear_2.weight"];
    void* tmp_linear2_bias   = weights[prefix + "linear_2.bias"];


    weight_loader_manager_glob->doCudaMemcpy(linear1_weight, tmp_linear1_weight, sizeof(T) * input_dim_ * output_dim_0_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(linear1_bias, tmp_linear1_bias, sizeof(T) * output_dim_0_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(linear2_weight, tmp_linear2_weight, sizeof(T) * output_dim_0_ * output_dim_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(linear2_bias, tmp_linear2_bias, sizeof(T) * output_dim_, memcpyKind);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(linear1_weight_h,
                        linear1_weight,
                        sizeof(T) * input_dim_ * output_dim_0_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);

        cudaMemcpyAsync(linear2_weight_h,
                        linear2_weight,
                        sizeof(T) * output_dim_0_ * output_dim_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template class TimestepEmbeddingBlockWeight<float>;
template class TimestepEmbeddingBlockWeight<half>;
#ifdef ENABLE_BF16
template class TimestepEmbeddingBlockWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff
