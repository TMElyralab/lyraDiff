#include "FluxAttnPostInt4ProcessorWeight.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxAttnPostInt4ProcessorWeight<T>::FluxAttnPostInt4ProcessorWeight(size_t        embedding_dim,
                                                                    LyraQuantType quant_level,
                                                                    IAllocator*   allocator)
{
    embedding_dim_     = embedding_dim;
    this->allocator_   = allocator;
    this->quant_level_ = quant_level;
    // cout << "embedding_dim " << embedding_dim << endl;
    gelu_proj_weight = new W4A4GemmWeight<T>(embedding_dim_ * 4, embedding_dim_, 32, true);
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FluxAttnPostInt4ProcessorWeight<T>::~FluxAttnPostInt4ProcessorWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        // deviceFree(gelu_proj_weight);
        // deviceFree(gelu_proj_bias);
        // deviceFree(gelu_proj_weight_scale);
        // deviceFree(gelu_proj_input_scale);

        deviceFree(ff_linear_weight);
        deviceFree(ff_linear_bias);
        deviceFree(ff_linear_weight_scale);
        deviceFree(ff_linear_input_scale);

        // free(gelu_proj_weight_h);
        free(ff_linear_weight_h);

        delete gelu_proj_weight;
    }
}

template<typename T>
void FluxAttnPostInt4ProcessorWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        // deviceMalloc(&gelu_proj_weight, embedding_dim_ * embedding_dim_ * 4);
        // deviceMalloc(&gelu_proj_bias, embedding_dim_ * 4);
        // deviceMalloc(&gelu_proj_weight_scale, 1);
        // deviceMalloc(&gelu_proj_input_scale, 1);

        deviceMalloc(&ff_linear_weight, embedding_dim_ * embedding_dim_ * 4);
        deviceMalloc(&ff_linear_bias, embedding_dim_);
        deviceMalloc(&ff_linear_weight_scale, 1);
        deviceMalloc(&ff_linear_input_scale, 1);

        ff_linear_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);
        // gelu_proj_weight_h = (T*)malloc(sizeof(T) * embedding_dim_ * embedding_dim_ * 4);
        gelu_proj_weight->mallocWeights();
        // this->lora_weight_map = {
        //     {"net.0.proj",
        //      new FP8LoraWeight<T>(
        //          {embedding_dim_ * embedding_dim_ * 4}, gelu_proj_weight, gelu_proj_weight_scale,
        //          gelu_proj_weight_h)},
        //     {"net.2",
        //      new FP8LoraWeight<T>(
        //          {embedding_dim_ * embedding_dim_ * 4}, ff_linear_weight, ff_linear_weight_scale,
        //          ff_linear_weight_h)}};
    }

    is_maintain_buffer = true;
}

template<typename T>
void FluxAttnPostInt4ProcessorWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    size_t cur_size     = 0;
    size_t cur_size_2   = 0;
    size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 4;
    T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size, false);

    // cur_size   = embedding_dim_ * embedding_dim_ * 4;
    // cur_size_2 = embedding_dim_ * 4;
    // MACROLoadFP8WeightFromBin(gelu_proj, cur_size, cur_size_2, "net.0.proj");
    cur_size   = embedding_dim_ * embedding_dim_ * 4;
    cur_size_2 = embedding_dim_;
    MACROLoadFP8WeightFromBin(ff_linear, cur_size, cur_size_2, "net.2");

    gelu_proj_weight->loadWeights(prefix + "net.0.proj.", model_file_type);

    cout << "FluxAttnPostInt4ProcessorWeight load weights" << endl;
}

template<typename T>
void FluxAttnPostInt4ProcessorWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                              std::unordered_map<std::string, void*>& weights,
                                                              cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // size_t cur_size     = 0;
    // size_t cur_size_2   = 0;
    // size_t cur_mem_size = sizeof(T) * embedding_dim_ * embedding_dim_ * 4;
    // T*     tmp_buffer   = (T*)this->allocator_->reMallocWithName("global_shared_weight_buffer_", cur_mem_size,
    // false);

    // cur_size   = embedding_dim_ * embedding_dim_ * 4;
    // cur_size_2 = embedding_dim_ * 4;
    // MACROLoadFP8WeightFromCache(gelu_proj, cur_size, cur_size_2, "net.0.proj");
    // cur_size   = embedding_dim_ * embedding_dim_ * 4;
    // cur_size_2 = embedding_dim_;
    // MACROLoadFP8WeightFromCache(ff_linear, cur_size, cur_size_2, "net.2");
}

template class FluxAttnPostInt4ProcessorWeight<float>;
template class FluxAttnPostInt4ProcessorWeight<half>;

#ifdef ENABLE_BF16
template class FluxAttnPostInt4ProcessorWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff