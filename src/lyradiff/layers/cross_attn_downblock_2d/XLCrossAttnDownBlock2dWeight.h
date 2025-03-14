#pragma once

#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {
template<typename T>
class XLCrossAttnDownBlock2dWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t temb_channels_;
    size_t head_num_;
    size_t norm_num_groups_;
    size_t cross_attn_dim_;
    size_t inner_transblock_num_;

    size_t downsampler_weight_size_;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    bool is_downsampler_              = true;
    T*   downsampler_weight           = nullptr;
    T*   downsampler_bias             = nullptr;
    T*   downsampler_weight_lora_buf_ = nullptr;

    XLTransformer2dBlockWeight<T>* xltransformer2d_block_weight1 = nullptr;
    XLTransformer2dBlockWeight<T>* xltransformer2d_block_weight2 = nullptr;
    Resnet2DBlockWeight<T>*        resnet_2d_block_weight1       = nullptr;
    Resnet2DBlockWeight<T>*        resnet_2d_block_weight2       = nullptr;

    XLCrossAttnDownBlock2dWeight() = default;
    XLCrossAttnDownBlock2dWeight(const size_t  in_channels,
                                 const size_t  out_channels,
                                 const size_t  temb_channels,
                                 const size_t  head_num,
                                 const size_t  cross_attn_dim,
                                 const size_t  norm_num_groups,
                                 const size_t  inner_transblock_num,
                                 bool          is_downsampler,
                                 LyraQuantType quant_level = LyraQuantType::NONE,
                                 IAllocator*   allocator   = nullptr);

    ~XLCrossAttnDownBlock2dWeight();
    virtual void mallocLoraBuffer();
    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
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
