#pragma once
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {

template<typename T>
class UNetMidBlock2DCrossAttnWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    // 0: resnet_0
    size_t in_channels_;

    // 1: attn
    size_t num_head_;
    size_t dim_per_head_;
    size_t group_num_;
    size_t encoder_hidden_dim_;

    // 2: resnet_1
    size_t out_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    Resnet2DBlockWeight<T>*      resnet_0_weights_;
    Transformer2dBlockWeight<T>* attn_weights_ = nullptr;
    Resnet2DBlockWeight<T>*      resnet_1_weights_;

    UNetMidBlock2DCrossAttnWeight() = default;
    UNetMidBlock2DCrossAttnWeight(const size_t        in_channels,
                                  const size_t        num_head,
                                  const size_t        group_num,
                                  const size_t        encoder_hidden_dim,
                                  const size_t        out_channels,
                                  const LyraQuantType quant_level = LyraQuantType::NONE,
                                  IAllocator*         allocator   = nullptr);

    ~UNetMidBlock2DCrossAttnWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
    virtual void loadLoraFromWeight(std::string                          lora_path,
                                    std::string                          prefix,
                                    std::unordered_map<std::string, T*>& lora_weights,
                                    float                                lora_alpha,
                                    FtCudaDataType                       lora_file_type,
                                    cudaStream_t                         stream);
    virtual void loadLoraFromCache(std::string                          prefix,
                                   std::unordered_map<std::string, T*>& lora_weights,
                                   float                                lora_alpha,
                                   bool                                 from_outside = true);
};

}  // namespace lyradiff
