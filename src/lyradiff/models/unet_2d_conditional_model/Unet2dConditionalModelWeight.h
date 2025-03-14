#pragma once
#include "src/lyradiff/layers/cross_attn_downblock_2d/CrossAttnDownBlock2dWeight.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/layers/down_block_2d/DownBlock2DWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/UNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/layers/upblock2d/UpBlock2dWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>
#include <vector>

namespace lyradiff {
template<typename T>
class Unet2dConditionalModelWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    std::vector<size_t> block_out_channels_ = {320, 640, 1280, 1280};
    size_t              temb_channels_      = 1280;
    size_t              cross_attn_dim_     = 768;
    size_t              head_num_           = 8;
    size_t              norm_num_groups_    = 32;
    size_t              input_channels_     = 4;
    size_t              output_channels_    = 4;

protected:
    bool is_maintain_buffer = false;
    bool is_malloced        = false;

public:
    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    T* conv_out_weight = nullptr;
    T* conv_out_bias   = nullptr;

    T* conv_out_norm_gamma = nullptr;
    T* conv_out_norm_beta  = nullptr;

    TimestepEmbeddingBlockWeight<T>*    timestep_embedding_weight;
    CrossAttnDownBlock2dWeight<T>*      cross_attn_down_block_2d_weight1;
    CrossAttnDownBlock2dWeight<T>*      cross_attn_down_block_2d_weight2;
    CrossAttnDownBlock2dWeight<T>*      cross_attn_down_block_2d_weight3;
    DownBlock2DWeight<T>*               down_block_2d_weight;
    UNetMidBlock2DCrossAttnWeight<T>*   mid_block_2d_weight;
    UpBlock2dWeight<T>*                 up_block_2d_weight;
    CrossAttnUpBlock2dWeight<T, true>*  cross_attn_up_block_2d_weight1;
    CrossAttnUpBlock2dWeight<T, true>*  cross_attn_up_block_2d_weight2;
    CrossAttnUpBlock2dWeight<T, false>* cross_attn_up_block_2d_weight3;

    Unet2dConditionalModelWeight(size_t              input_channels  = 4,
                                 size_t              output_channels = 4,
                                 const LyraQuantType quant_level     = LyraQuantType::NONE,
                                 IAllocator*         allocator       = nullptr, const std::string &sd_ver = "sd15");

    ~Unet2dConditionalModelWeight();

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
    
    virtual void loadS3DiffLoraFromStateDict(std::unordered_map<std::string, T*>& lora_weights, bool is_alpha);
};

}  // namespace lyradiff
