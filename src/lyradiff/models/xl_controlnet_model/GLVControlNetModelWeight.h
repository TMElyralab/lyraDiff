#pragma once
#include "src/lyradiff/layers/cross_attn_downblock_2d/XLCrossAttnDownBlock2dWeight.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/XLCrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/layers/down_block_2d/XLDownBlock2DWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlockWeight.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/XLUNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/layers/upblock2d/XLUpBlock2dWeight.h"

#include "src/lyradiff/utils/cuda_utils.h"
#include <vector>

namespace lyradiff {
template<typename T>
class GLVControlNetModelWeight {
private:
    std::vector<size_t> block_out_channels_ = {320, 640, 1280};
    std::vector<size_t> head_nums_          = {10, 20};
    std::vector<size_t> inner_trans_nums_   = {2, 10};
    size_t              temb_channels_      = 1280;
    size_t              cross_attn_dim_     = 2048;
    size_t              norm_num_groups_    = 32;
    size_t              input_channels_     = 4;
    size_t              add_emb_input_dim_  = 2816;
    const bool          use_runtime_augemb_ = false;

    size_t controlnet_condition_channels_ = 3;

protected:
    bool is_maintain_buffer = false;
    bool is_malloced        = false;

public:
    T* conv_in_weight = nullptr;
    T* conv_in_bias   = nullptr;

    T* input_hint_conv_weight = nullptr;
    T* input_hint_conv_bias   = nullptr;

    TextTimeEmbeddingBlockWeight<T>    texttime_embedding_weight;
    TimestepEmbeddingBlockWeight<T>    timestep_embedding_weight;
    XLDownBlock2DWeight<T>             down_block_2d_weight;
    XLCrossAttnDownBlock2dWeight<T>    cross_attn_down_block_2d_weight1;
    XLCrossAttnDownBlock2dWeight<T>    cross_attn_down_block_2d_weight2;
    XLUNetMidBlock2DCrossAttnWeight<T> mid_block_2d_weight;

    GLVControlNetModelWeight(const bool use_runtime_augemb = false, LyraQuantType quant_level = LyraQuantType::NONE);

    ~GLVControlNetModelWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
