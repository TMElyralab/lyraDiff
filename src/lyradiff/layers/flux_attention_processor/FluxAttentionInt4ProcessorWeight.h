#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/flux_attention_processor/FluxAttentionProcessorWeight.h"
#include "src/lyradiff/layers/w4a4_gemm_block/W4A4GemmWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxAttentionInt4ProcessorWeight: public FluxAttentionProcessorWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    // __nv_fp8_e4m3* to_qkv_weight;
    // T*             to_qkv_weight_h;
    // T*             to_qkv_bias;
    // float*         to_qkv_weight_scale;
    // float*         to_qkv_input_scale;

    // __nv_fp8_e4m3* encoder_to_qkv_weight;
    // T*             encoder_to_qkv_weight_h;
    // T*             encoder_to_qkv_bias;
    // float*         encoder_to_qkv_weight_scale;
    // float*         encoder_to_qkv_input_scale;

    T* qk_norm_weight;
    T* encoder_qk_norm_weight;

    W4A4GemmWeight<T>* to_qkv_weight;
    W4A4GemmWeight<T>* encoder_to_qkv_weight;
    W4A4GemmWeight<T>* to_out_weight;
    W4A4GemmWeight<T>* encoder_to_out_weight;

    // __nv_fp8_e4m3* to_out_weight;
    // T*             to_out_weight_h;
    // T*             to_out_bias;
    // float*         to_out_weight_scale;
    // float*         to_out_input_scale;

    // __nv_fp8_e4m3* encoder_to_out_weight;
    // T*             encoder_to_out_weight_h;
    // T*             encoder_to_out_bias;
    // float*         encoder_to_out_weight_scale;
    // float*         encoder_to_out_input_scale;

    FluxAttentionInt4ProcessorWeight() = default;
    FluxAttentionInt4ProcessorWeight(size_t        embedding_dim,
                                     size_t        embedding_head_num,
                                     size_t        embedding_head_dim,
                                     LyraQuantType quant_level,
                                     IAllocator*   allocator);

    virtual ~FluxAttentionInt4ProcessorWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
