#pragma once

#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlockWeight.h"
#include "src/lyradiff/layers/zero_cross_attn/ZeroCrossAttnWeight.h"
#include "src/lyradiff/layers/zero_sft/ZeroSFTWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>

namespace lyradiff {
template<typename T>
class GLVCrossAttnUpBlock2dWeight: public IFCBasicTransformerContainerWeight<T> {

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t prev_output_channel_;
    size_t temb_channels_;
    size_t head_num_;
    size_t norm_num_groups_;
    size_t cross_attn_dim_;
    size_t inner_trans_num_;
    size_t upsampler_weight_size_;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    T* upsampler_weight           = nullptr;
    T* upsampler_bias             = nullptr;
    T* upsampler_weight_lora_buf_ = nullptr;

    bool is_maintain_project_buffer = false;

    XLTransformer2dBlockWeight<T>* xltransformer2d_block_weight1 = nullptr;
    XLTransformer2dBlockWeight<T>* xltransformer2d_block_weight2 = nullptr;
    XLTransformer2dBlockWeight<T>* xltransformer2d_block_weight3 = nullptr;

    Resnet2DBlockWeight<T>* resnet_2d_block_weight1 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight2 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight3 = nullptr;

    ZeroSFTWeight<T>* project_module_weight1 = nullptr;
    ZeroSFTWeight<T>* project_module_weight2 = nullptr;
    ZeroSFTWeight<T>* project_module_weight3 = nullptr;

    ZeroCrossAttnWeight<T>* cross_project_module_weight = nullptr;

    GLVCrossAttnUpBlock2dWeight() = default;
    GLVCrossAttnUpBlock2dWeight(const size_t  in_channels,
                                const size_t  out_channels,
                                const size_t  prev_output_channel,
                                const size_t  temb_channels,
                                const size_t  head_num,
                                const size_t  cross_attn_dim,
                                const size_t  norm_num_groups,
                                const size_t  inner_trans_num,
                                LyraQuantType quant_level = LyraQuantType::NONE,
                                IAllocator*   allocator   = nullptr);

    ~GLVCrossAttnUpBlock2dWeight();

    virtual void loadProjectWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadProjectWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind);
    virtual void mallocProjectModuleWeights();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
    virtual void mallocLoraBuffer();
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
