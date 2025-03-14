#include "FluxAttnPostProcessorWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttnPostProcessorWeight<T>::FluxAttnPostProcessorWeight(size_t        embedding_dim,
                                                            LyraQuantType quant_level,
                                                            IAllocator*   allocator)
{
    embedding_dim_     = embedding_dim;
    this->allocator_   = allocator;
    this->quant_level_ = quant_level;
    // cout << "embedding_dim " << embedding_dim << endl;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxAttnPostProcessorWeight<T>::~FluxAttnPostProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(gelu_proj_weight);
        deviceFree(gelu_proj_bias);

        deviceFree(ff_linear_weight);
        deviceFree(ff_linear_bias);
        if (this->quant_level_ != LyraQuantType::NONE) {
            free(ff_linear_weight_h);
            free(gelu_proj_weight_h);
        }
    }
}

template<typename T>
void FluxAttnPostProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&gelu_proj_weight, embedding_dim_ * embedding_dim_ * 4);
        deviceMalloc(&gelu_proj_bias, embedding_dim_ * 4);

        deviceMalloc(&ff_linear_weight, embedding_dim_ * embedding_dim_ * 4);
        deviceMalloc(&ff_linear_bias, embedding_dim_);

        // this->lora_weight_size_map = {{"net.0.proj", embedding_dim_ * embedding_dim_ * 4},
        //                               {"net.2", embedding_dim_ * embedding_dim_ * 4}};
        // this->lora_weight_map      = {{"net.0.proj", gelu_proj_weight}, {"net.2", ff_linear_weight}};

        if (this->quant_level_ != LyraQuantType::NONE) {
            ff_linear_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);
            gelu_proj_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);

            this->lora_weight_map = {
                {"net.0.proj",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_ * 4}, gelu_proj_weight, gelu_proj_weight_h)},
                {"net.2",
                 new LoraWeightV2<T>({embedding_dim_ * embedding_dim_ * 4}, ff_linear_weight, ff_linear_weight_h)}};
        }
        else {
            this->lora_weight_map = {
                {"net.0.proj", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * 4}, gelu_proj_weight)},
                {"net.2", new LoraWeight<T>({embedding_dim_ * embedding_dim_ * 4}, ff_linear_weight)}};
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttnPostProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    loadWeightFromBin<T>(
        gelu_proj_weight, {embedding_dim_ * embedding_dim_ * 4}, prefix + "net.0.proj.weight.bin", model_file_type);
    loadWeightFromBin<T>(gelu_proj_bias, {embedding_dim_ * 4}, prefix + "net.0.proj.bias.bin", model_file_type);

    loadWeightFromBin<T>(
        ff_linear_weight, {embedding_dim_ * embedding_dim_ * 4}, prefix + "net.2.weight.bin", model_file_type);
    loadWeightFromBin<T>(ff_linear_bias, {embedding_dim_}, prefix + "net.2.bias.bin", model_file_type);

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpy(gelu_proj_weight_h,
                   gelu_proj_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * 4,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(ff_linear_weight_h,
                   ff_linear_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * 4,
                   cudaMemcpyDeviceToHost);
    }
}

template<typename T>
void FluxAttnPostProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                          std::unordered_map<std::string, void*>& weights,
                                                          cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_gelu_weight = weights[prefix + "net.0.proj.weight"];
    void* tmp_gelu_bias   = weights[prefix + "net.0.proj.bias"];

    void* tmp_ff_linear_weight = weights[prefix + "net.2.weight"];
    void* tmp_ff_linear_bias   = weights[prefix + "net.2.bias"];

    check_cuda_error(
        cudaMemcpy(gelu_proj_weight, tmp_gelu_weight, sizeof(T) * embedding_dim_ * embedding_dim_ * 4, memcpyKind));
    check_cuda_error(cudaMemcpy(gelu_proj_bias, tmp_gelu_bias, sizeof(T) * embedding_dim_ * 4, memcpyKind));

    check_cuda_error(cudaMemcpy(
        ff_linear_weight, tmp_ff_linear_weight, sizeof(T) * embedding_dim_ * embedding_dim_ * 4, memcpyKind));
    check_cuda_error(cudaMemcpy(ff_linear_bias, tmp_ff_linear_bias, sizeof(T) * embedding_dim_, memcpyKind));

    if (this->quant_level_ != LyraQuantType::NONE) {
        cudaMemcpy(gelu_proj_weight_h,
                   gelu_proj_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * 4,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(ff_linear_weight_h,
                   ff_linear_weight,
                   sizeof(T) * embedding_dim_ * embedding_dim_ * 4,
                   cudaMemcpyDeviceToHost);
    }
}

template class FluxAttnPostProcessorWeight<float>;
template class FluxAttnPostProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttnPostProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff