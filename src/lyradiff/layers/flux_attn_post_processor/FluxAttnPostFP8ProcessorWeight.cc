#include "FluxAttnPostFP8ProcessorWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttnPostFP8ProcessorWeight<T>::FluxAttnPostFP8ProcessorWeight(size_t        embedding_dim,
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
FluxAttnPostFP8ProcessorWeight<T>::~FluxAttnPostFP8ProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(gelu_proj_weight);
        deviceFree(gelu_proj_bias);
        deviceFree(gelu_proj_weight_scale);
        deviceFree(gelu_proj_input_scale);

        deviceFree(ff_linear_weight);
        deviceFree(ff_linear_bias);
        deviceFree(ff_linear_weight_scale);
        deviceFree(ff_linear_input_scale);

        free(gelu_proj_weight_h);
        free(ff_linear_weight_h);
    }
}

template<typename T>
void FluxAttnPostFP8ProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&gelu_proj_weight, embedding_dim_ * embedding_dim_ * 4);
        deviceMalloc(&gelu_proj_bias, embedding_dim_ * 4);
        deviceMalloc(&gelu_proj_weight_scale, 1);
        deviceMalloc(&gelu_proj_input_scale, 1);

        deviceMalloc(&ff_linear_weight, embedding_dim_ * embedding_dim_ * 4);
        deviceMalloc(&ff_linear_bias, embedding_dim_);
        deviceMalloc(&ff_linear_weight_scale, 1);
        deviceMalloc(&ff_linear_input_scale, 1);

        ff_linear_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);
        gelu_proj_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);

        this->lora_weight_map = {
            {"net.0.proj",
             new FP8LoraWeight<T>(
                 {embedding_dim_ * embedding_dim_ * 4}, gelu_proj_weight, gelu_proj_weight_scale, gelu_proj_weight_h)},
            {"net.2",
             new FP8LoraWeight<T>(
                 {embedding_dim_ * embedding_dim_ * 4}, ff_linear_weight, ff_linear_weight_scale, ff_linear_weight_h)}};
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttnPostFP8ProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 4;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 4;
    cur_size_2 = embedding_dim_ * 4;
    MACROLoadFP8WeightFromBin(gelu_proj, cur_size, cur_size_2, "net.0.proj");
    cur_size   = embedding_dim_ * embedding_dim_ * 4;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromBin(ff_linear, cur_size, cur_size_2, "net.2");

    // loadWeightFromBin<__nv_fp8_e4m3>(
    //     gelu_proj_weight, {embedding_dim_ * embedding_dim_ * 4}, prefix + "net.0.proj_weight_fp8.bin",
    //     model_file_type);
    // loadWeightFromBin<T>(gelu_proj_bias, {embedding_dim_ * 4}, prefix + "net.0.proj.bias.bin", model_file_type);
    // loadWeightFromBin<float>(gelu_proj_weight_scale, {1}, prefix + "net.0.proj_weight_scale.bin",
    // FtCudaDataType::FP32); loadWeightFromBin<float>(gelu_proj_input_scale, {1}, prefix +
    // "net.0.proj_input_scale.bin", FtCudaDataType::FP32);

    // loadWeightFromBin<__nv_fp8_e4m3>(
    //     ff_linear_weight, {embedding_dim_ * embedding_dim_ * 4}, prefix + "net.2_weight_fp8.bin", model_file_type);
    // loadWeightFromBin<T>(ff_linear_bias, {embedding_dim_}, prefix + "net.2.bias.bin", model_file_type);
    // loadWeightFromBin<float>(ff_linear_weight_scale, {1}, prefix + "net.2_weight_scale.bin", FtCudaDataType::FP32);
    // loadWeightFromBin<float>(ff_linear_input_scale, {1}, prefix + "net.2_input_scale.bin", FtCudaDataType::FP32);
}

template<typename T>
void FluxAttnPostFP8ProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                             std::unordered_map<std::string, void*>& weights,
                                                             cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 4;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    cur_size   = embedding_dim_ * embedding_dim_ * 4;
    cur_size_2 = embedding_dim_ * 4;
    MACROLoadFP8WeightFromCache(gelu_proj, cur_size, cur_size_2, "net.0.proj");
    cur_size   = embedding_dim_ * embedding_dim_ * 4;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromCache(ff_linear, cur_size, cur_size_2, "net.2");
}

template class FluxAttnPostFP8ProcessorWeight<float>;
template class FluxAttnPostFP8ProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttnPostFP8ProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff