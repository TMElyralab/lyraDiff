#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class XLDownBlock2DWeight: public IFLoraWeight<T> {
private:
    size_t in_channels_;
    size_t inter_channels_;
    size_t out_channels_;
    bool   is_downsampler_;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    T* downsampler_weight = nullptr;
    T* downsampler_bias   = nullptr;

    T* downsampler_weight_lora_buf = nullptr;

    Resnet2DBlockWeight<T>* resnet_0_weights = nullptr;
    Resnet2DBlockWeight<T>* resnet_1_weights = nullptr;

    XLDownBlock2DWeight() = default;
    XLDownBlock2DWeight(const size_t in_channels_,
                        const size_t inter_channels_,
                        const size_t out_channels_,
                        const bool   is_downsampler_,
                        IAllocator*  allocator = nullptr);

    ~XLDownBlock2DWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
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
