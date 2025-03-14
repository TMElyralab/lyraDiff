#include "AdaLayerNormWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
AdaLayerNormWeight<T>::AdaLayerNormWeight(size_t        embedding_dim,
                                          size_t        embedding_scale,
                                          LyraQuantType quant_level,
                                          IAllocator*   allocator)
{
    embedding_dim_     = embedding_dim;
    embedding_scale_   = embedding_scale;
    this->allocator_   = allocator;
    this->quant_level_ = quant_level;
    // cout << "embedding_dim " << embedding_dim << endl;
    // cout << "embedding_scale " << embedding_scale << endl;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
AdaLayerNormWeight<T>::~AdaLayerNormWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(linear_weight);
        deviceFree(linear_bias);

        if (this->quant_level_ != LyraQuantType::NONE) {
            free(linear_weight_h);
        }
    }
}

template<typename T>
void AdaLayerNormWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&linear_weight, embedding_dim_ * embedding_dim_ * embedding_scale_);
        deviceMalloc(&linear_bias, embedding_dim_ * embedding_scale_);

        if (this->quant_level_ != LyraQuantType::NONE) {
            linear_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * embedding_scale_);

            this->lora_weight_map = {{"linear",
                                      new LoraWeightV2<T>({embedding_dim_ * embedding_dim_ * embedding_scale_},
                                                          linear_weight,
                                                          linear_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"linear", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * embedding_scale_}, linear_weight)}};
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void AdaLayerNormWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // cout << "AdaLayerNormWeight<T>::loadWeights prefix: " << prefix << endl;
    // cout << "AdaLayerNormWeight linear_weight size : " << embedding_dim_ * embedding_dim_ * embedding_scale_ << endl;
    // cout << "AdaLayerNormWeight linear_bias size : " << embedding_dim_ * embedding_scale_ << endl;

    loadWeightFromBin<T>(linear_weight,
                         {embedding_dim_ * embedding_dim_ * embedding_scale_},
                         prefix + "linear.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(linear_bias, {embedding_dim_ * embedding_scale_}, prefix + "linear.bias.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpyAsync(linear_weight_h,
                        linear_weight,
                        sizeof(T) * embedding_dim_ * embedding_dim_ * embedding_scale_,
                        cudaMemcpyDeviceToHost,
                        cudaStreamDefault);
    }
}

template<typename T>
void AdaLayerNormWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                 std::unordered_map<std::string, void*>& weights,
                                                 cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_linear_weight = weights[prefix + "linear.weight"];
    void* tmp_linear_bias   = weights[prefix + "linear.bias"];

    check_cuda_error(cudaMemcpy(
        linear_weight, tmp_linear_weight, sizeof(T) * embedding_dim_ * embedding_dim_ * embedding_scale_, memcpyKind));
    check_cuda_error(
        cudaMemcpy(linear_bias, tmp_linear_bias, sizeof(T) * embedding_dim_ * embedding_scale_, memcpyKind));

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpy(linear_weight_h,
                   linear_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * embedding_scale_,
                   cudaMemcpyDeviceToHost);
    }
}

template class AdaLayerNormWeight<float>;
template class AdaLayerNormWeight<half>;

#ifdef ENABLE_BF16
template class AdaLayerNormWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff