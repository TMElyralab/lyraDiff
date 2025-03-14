#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class DownBlock2DWeight: public IFLoraWeight<T> {
private:
    size_t in_channels_;
    size_t inter_channels_;
    size_t out_channels_;

protected:
    bool is_maintain_buffer = false;

public:
    Resnet2DBlockWeight<T>* resnet_0_weights;
    Resnet2DBlockWeight<T>* resnet_1_weights;

    DownBlock2DWeight() = default;
    DownBlock2DWeight(const size_t in_channels_,
                      const size_t inter_channels_,
                      const size_t out_channels_,
                      IAllocator*  allocator = nullptr);

    ~DownBlock2DWeight();

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
