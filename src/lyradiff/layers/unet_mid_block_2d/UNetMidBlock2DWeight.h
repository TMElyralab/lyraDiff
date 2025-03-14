#pragma once
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {

template<typename T>
class UNetMidBlock2DWeight {
private:
    // fields
    size_t in_channels_;
    size_t num_head_;
    size_t dim_per_head_;
    size_t group_num_;
    size_t temb_channels_;

    // weight buf size
    size_t attention_qkv_size_;
    size_t attention_to_out_size_;

protected:
    bool is_maintain_buffer = false;

public:
    Resnet2DBlockWeight<T>* resnet_0_weights_;
    Resnet2DBlockWeight<T>* resnet_1_weights_;

    T* gnorm_gamma = nullptr;
    T* gnorm_beta  = nullptr;

    T* attention_qkv_weight    = nullptr;
    T* attention_qkv_bias      = nullptr;
    T* attention_to_out_weight = nullptr;
    T* attention_to_out_bias   = nullptr;

    UNetMidBlock2DWeight() = default;
    UNetMidBlock2DWeight(const size_t in_channels,
                         const size_t num_head,
                         const size_t group_num,
                         const size_t temb_channels);

    ~UNetMidBlock2DWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
