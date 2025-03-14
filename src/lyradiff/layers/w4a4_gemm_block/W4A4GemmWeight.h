#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class W4A4GemmWeight: public IFLoraWeight<T> {
private:
    size_t d_out_;
    size_t d_in_;
    size_t lora_rank_  = 32;
    size_t group_size_ = 64;
    size_t group_num_  = 0;
    bool   has_bias_   = true;

protected:
    bool is_maintain_buffer = false;

public:
    int8_t* packed_weight;      // int8 * d_out * d_in / 2
    float*  weight_scale;       // fp32 * d_out * 48
    T*      lora_up;            // bf16 * d_out * 32
    T*      lora_down;          // bf16 * d_in * 32
    T*      bias;               // bf16 * d_out
    float*  smooth;             // fp32 * d_in

    W4A4GemmWeight() = default;
    W4A4GemmWeight(size_t        d_out,
                   size_t        d_in,
                   size_t        lora_rank   = 32,
                   bool          has_bias    = false,
                   size_t        group_size  = 64,
                   LyraQuantType quant_level = LyraQuantType::NONE,
                   IAllocator*   allocator   = nullptr);

    virtual ~W4A4GemmWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
