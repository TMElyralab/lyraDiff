#pragma once
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class ZeroCrossAttnWeight {
private:
    size_t query_dim_;
    size_t context_dim_;
    size_t heads_;
    size_t dim_head_ = 64;

    size_t inner_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    T* attention_q_weight      = nullptr;
    T* attention_kv_weight     = nullptr;
    T* attention_to_out_weight = nullptr;
    T* attention_to_out_bias   = nullptr;

    T* norm1_gamma = nullptr;
    T* norm1_beta  = nullptr;

    T* norm2_gamma = nullptr;
    T* norm2_beta  = nullptr;

    ZeroCrossAttnWeight() = default;
    ZeroCrossAttnWeight(size_t query_dim, size_t context_dim);

    ~ZeroCrossAttnWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
