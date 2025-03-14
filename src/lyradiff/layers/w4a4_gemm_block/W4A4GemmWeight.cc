#include "W4A4GemmWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
W4A4GemmWeight<T>::W4A4GemmWeight(size_t        d_out,
                                  size_t        d_in,
                                  size_t        lora_rank,
                                  bool          has_bias,
                                  size_t        group_size,
                                  LyraQuantType quant_level,
                                  IAllocator*   allocator)
{
    d_out_      = d_out;
    d_in_       = d_in;
    lora_rank_  = lora_rank;
    has_bias_   = has_bias;
    group_size_ = group_size;
    group_num_  = d_in_ / group_size_;

    // this->allocator_   = allocator;
    // this->quant_level_ = quant_level;

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
W4A4GemmWeight<T>::~W4A4GemmWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(packed_weight);
        deviceFree(weight_scale);
        deviceFree(lora_up);
        deviceFree(lora_down);
        deviceFree(smooth);
        if (has_bias_) {
            deviceFree(bias);
        }
    }
}

template<typename T>
void W4A4GemmWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&packed_weight, d_out_ * d_in_ / 2);
        deviceMalloc(&weight_scale, d_out_ * group_num_);
        deviceMalloc(&lora_up, d_out_ * lora_rank_);
        deviceMalloc(&lora_down, d_in_ * lora_rank_);
        deviceMalloc(&smooth, d_in_);

        if (has_bias_) {
            deviceMalloc(&bias, d_out_);
        }
    }

    is_maintain_buffer = true;
}

template<typename T>
void W4A4GemmWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
    // n * k / 4 + (k + n) * 32 + k * 2 + n * 2

    loadWeightFromBin<int8_t>(packed_weight, {d_out_ * d_in_ / 2}, prefix + "packed_weight.bin", FtCudaDataType::INT8);
    loadWeightFromBin<float>(weight_scale, {d_out_ * group_num_}, prefix + "weight_scale.bin", FtCudaDataType::FP32);
    loadWeightFromBin<T>(lora_up, {d_out_ * lora_rank_}, prefix + "lora_up.bin", model_file_type);
    loadWeightFromBin<T>(lora_down, {d_in_ * lora_rank_}, prefix + "lora_down.bin", model_file_type);
    loadWeightFromBin<float>(smooth, {d_in_}, prefix + "smooth.bin", FtCudaDataType::FP32);

    if (has_bias_) {
        loadWeightFromBin<T>(bias, {d_out_}, prefix + "bias.bin", model_file_type);
    }
}

template<typename T>
void W4A4GemmWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }
}

template class W4A4GemmWeight<float>;
template class W4A4GemmWeight<half>;

#ifdef ENABLE_BF16
template class W4A4GemmWeight<__nv_bfloat16>;
#endif
}  // namespace lyradiff