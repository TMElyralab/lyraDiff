#include "AdaFP8LayerNormWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
AdaFP8LayerNormWeight<T>::AdaFP8LayerNormWeight(size_t        embedding_dim,
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
AdaFP8LayerNormWeight<T>::~AdaFP8LayerNormWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(linear_weight);
        deviceFree(linear_bias);
        deviceFree(linear_weight_scale);
        deviceFree(linear_input_scale);
    }
}

template<typename T>
void AdaFP8LayerNormWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&linear_weight, embedding_dim_ * embedding_dim_ * embedding_scale_);
        deviceMalloc(&linear_bias, embedding_dim_ * embedding_scale_);
        deviceMalloc(&linear_weight_scale, 1);
        deviceMalloc(&linear_input_scale, 1);

        linear_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * embedding_scale_);

        this->lora_weight_map = {{"linear",
                                  new FP8LoraWeight<T>({embedding_dim_ * embedding_dim_ * embedding_scale_},
                                                       linear_weight,
                                                       linear_weight_scale,
                                                       linear_weight_h)}};
    }

    is_maintain_buffer = true;
}

template<typename T>
void AdaFP8LayerNormWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // cout << "AdaLayerNormWeight<T>::loadWeights prefix: " << prefix << endl;
    // cout << "AdaLayerNormWeight linear_weight size : " << embedding_dim_ * embedding_dim_ * embedding_scale_ << endl;
    // cout << "AdaLayerNormWeight linear_bias size : " << embedding_dim_ * embedding_scale_ << endl;
    size_t cur_size     = embedding_dim_ * embedding_dim_ * embedding_scale_;
    size_t cur_size_2   = embedding_dim_ * embedding_scale_;
    size_t cur_mem_size = sizeof(T) * cur_size;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    MACROLoadFP8WeightFromBin(linear, cur_size, cur_size_2, "linear");
}

template<typename T>
void AdaFP8LayerNormWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                    std::unordered_map<std::string, void*>& weights,
                                                    cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    size_t cur_size     = embedding_dim_ * embedding_dim_ * embedding_scale_;
    size_t cur_size_2   = embedding_dim_ * embedding_scale_;
    size_t cur_mem_size = sizeof(T) * cur_size;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    MACROLoadFP8WeightFromCache(linear, cur_size, cur_size_2, "linear");
}

template class AdaFP8LayerNormWeight<float>;
template class AdaFP8LayerNormWeight<half>;

#ifdef ENABLE_BF16
template class AdaFP8LayerNormWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff