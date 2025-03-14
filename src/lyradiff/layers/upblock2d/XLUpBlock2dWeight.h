#pragma once

#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class XLUpBlock2dWeight: public IFLoraWeight<T> {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t prev_output_channel_;
    size_t norm_num_groups_;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    Resnet2DBlockWeight<T>* resnet_2d_block_weight1 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight2 = nullptr;
    Resnet2DBlockWeight<T>* resnet_2d_block_weight3 = nullptr;

    XLUpBlock2dWeight() = default;
    XLUpBlock2dWeight(const size_t in_channels,
                      const size_t out_channels,
                      const size_t prev_output_channel,
                      const size_t norm_num_groups,
                      IAllocator*  allocator = nullptr);

    ~XLUpBlock2dWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void mallocLoraBuffer();
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
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

    virtual void mallocWeights();
};

}  // namespace lyradiff
