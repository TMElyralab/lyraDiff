#pragma once
#include "src/lyradiff/layers/controlnet_conditioning_embedding/ControlNetConditioningEmbeddingWeight.h"
#include "src/lyradiff/layers/controlnet_final_conv/ControlNetFinalConvWeight.h"
#include "src/lyradiff/layers/cross_attn_downblock_2d/CrossAttnDownBlock2dWeight.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/layers/down_block_2d/DownBlock2DWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/UNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/layers/upblock2d/UpBlock2dWeight.h"

#include "src/lyradiff/utils/cuda_utils.h"
#include <vector>

namespace lyradiff {
template<typename T>
class ControlNetModelWeight {
private:
    std::vector<size_t> block_out_channels_            = {320, 640, 1280, 1280};
    size_t              temb_channels_                 = 1280;
    size_t              cross_attn_dim_                = 768;
    size_t              head_num_                      = 8;
    size_t              norm_num_groups_               = 32;
    size_t              input_channels_                = 4;
    size_t              controlnet_condition_channels_ = 3;

protected:
    bool is_maintain_buffer = false;
    bool is_malloced        = false;

public:
    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    TimestepEmbeddingBlockWeight<T>*          timestep_embedding_weight;
    CrossAttnDownBlock2dWeight<T>*            cross_attn_down_block_2d_weight1;
    CrossAttnDownBlock2dWeight<T>*            cross_attn_down_block_2d_weight2;
    CrossAttnDownBlock2dWeight<T>*            cross_attn_down_block_2d_weight3;
    DownBlock2DWeight<T>*                     down_block_2d_weight;
    UNetMidBlock2DCrossAttnWeight<T>*         mid_block_2d_weight;
    ControlNetConditioningEmbeddingWeight<T>* controlnet_conditioning_embedding_weight;
    ControlNetFinalConvWeight<T>*             controlnet_final_conv_weight;

    ControlNetModelWeight(const LyraQuantType quant_level = LyraQuantType::NONE);

    ~ControlNetModelWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
