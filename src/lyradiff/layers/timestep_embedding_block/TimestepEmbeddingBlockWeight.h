#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class TimestepEmbeddingBlockWeight: public IFLoraWeight<T> {
private:
    size_t input_dim_;
    size_t output_dim_0_;
    size_t output_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    T* linear1_weight   = nullptr;
    T* linear1_weight_h = nullptr;
    T* linear1_bias     = nullptr;

    T* linear2_weight   = nullptr;
    T* linear2_weight_h = nullptr;
    T* linear2_bias     = nullptr;

    TimestepEmbeddingBlockWeight() = default;
    TimestepEmbeddingBlockWeight(const size_t        input_dim_,
                                 const size_t        output_dim_0_,
                                 const size_t        output_dim_,
                                 const LyraQuantType quant_level = LyraQuantType::NONE,
                                 IAllocator*         allocator   = nullptr);

    ~TimestepEmbeddingBlockWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
