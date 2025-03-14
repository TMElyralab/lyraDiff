#pragma once
#include "src/lyradiff/layers/cross_attn_downblock_2d/XLCrossAttnDownBlock2dWeight.h"
#include "src/lyradiff/layers/cross_attn_upblock_2d/GLVCrossAttnUpBlock2dWeight.h"
#include "src/lyradiff/layers/down_block_2d/XLDownBlock2DWeight.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlockWeight.h"
#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlockWeight.h"
#include "src/lyradiff/layers/unet_mid_block_2d_cross_attn/XLUNetMidBlock2DCrossAttnWeight.h"
#include "src/lyradiff/layers/upblock2d/GLVUpBlock2dWeight.h"
#include "src/lyradiff/layers/zero_sft/ZeroSFTWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <unordered_map>
#include <vector>

namespace lyradiff {
template<typename T>
class GLVUnet2dConditionalModelWeight: public IFCBasicTransformerContainerWeight<T> {
private:
    std::vector<size_t> block_out_channels_ = {320, 640, 1280};
    std::vector<size_t> head_nums_          = {10, 20};
    std::vector<size_t> inner_trans_nums_   = {2, 10};
    size_t              temb_channels_      = 1280;
    size_t              cross_attn_dim_     = 2048;
    size_t              norm_num_groups_    = 32;
    size_t              input_channels_     = 4;
    size_t              output_channels_    = 4;
    size_t              add_emb_input_dim_  = 2816;
    const bool          use_runtime_augemb_ = false;

    bool is_maintain_project_buffer = false;

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

    TextTimeEmbeddingBlockWeight<T>*    texttime_embedding_weight;
    TimestepEmbeddingBlockWeight<T>*    timestep_embedding_weight;
    XLDownBlock2DWeight<T>*             down_block_2d_weight;
    XLCrossAttnDownBlock2dWeight<T>*    cross_attn_down_block_2d_weight1;
    XLCrossAttnDownBlock2dWeight<T>*    cross_attn_down_block_2d_weight2;
    XLUNetMidBlock2DCrossAttnWeight<T>* mid_block_2d_weight;
    GLVCrossAttnUpBlock2dWeight<T>*     cross_attn_up_block_2d_weight1;
    GLVCrossAttnUpBlock2dWeight<T>*     cross_attn_up_block_2d_weight2;
    GLVUpBlock2dWeight<T>*              up_block_2d_weight;
    ZeroSFTWeight<T>*                   mid_block_project_module_weight;

    GLVUnet2dConditionalModelWeight(const size_t        input_channels     = 4,
                                    const size_t        output_channels    = 4,
                                    const bool          use_runtime_augemb = false,
                                    const LyraQuantType quant_level        = LyraQuantType::NONE);

    ~GLVUnet2dConditionalModelWeight();

    virtual void loadProjectWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadProjectWeightsFromCache(std::string                             prefix,
                                             std::unordered_map<std::string, void*>& weights,
                                             cudaMemcpyKind                          memcpyKind);

    virtual void mallocProjectModuleWeights();

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
