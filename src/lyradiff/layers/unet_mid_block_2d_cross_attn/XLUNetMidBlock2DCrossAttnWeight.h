#pragma once
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {

template<typename T>
class XLUNetMidBlock2DCrossAttnWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    // 0: resnet_0
    size_t in_channels_;

    // 1: attn
    size_t num_head_;
    size_t dim_per_head_;
    size_t group_num_;
    size_t encoder_hidden_dim_;
    size_t inner_trans_num_;

    // 2: resnet_1
    size_t out_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    Resnet2DBlockWeight<T>*        resnet_0_weights_;
    XLTransformer2dBlockWeight<T>* attn_weights_;
    Resnet2DBlockWeight<T>*        resnet_1_weights_;

    XLUNetMidBlock2DCrossAttnWeight() = default;
    XLUNetMidBlock2DCrossAttnWeight(const size_t  in_channels,
                                    const size_t  num_head,
                                    const size_t  group_num,
                                    const size_t  encoder_hidden_dim,
                                    const size_t  inner_trans_num,
                                    const size_t  out_channels,
                                    LyraQuantType quant_level = LyraQuantType::NONE,
                                    IAllocator*   allocator   = nullptr);

    // XLUNetMidBlock2DCrossAttnWeight(const size_t in_channels,
    //                                 const size_t num_head,
    //                                 const size_t group_num,
    //                                 const size_t encoder_hidden_dim,
    //                                 const size_t out_channels,
    //                                 size_t       quant_level = 0);

    ~XLUNetMidBlock2DCrossAttnWeight();

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
