#pragma once
#include "FluxSingleTransformerBlockWeight.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaFP8LayerNormWeight.h"
#include "src/lyradiff/layers/flux_single_attention_processor/FluxSingleAttentionFP8ProcessorWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxSingleTransformerFP8BlockWeight: public FluxSingleTransformerBlockWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;
    size_t mlp_scale_;

protected:
    bool is_maintain_buffer = false;

public:
    AdaLayerNormWeight<T>*                 ada_norm_weight;
    FluxSingleAttentionProcessorWeight<T>* attn_weight;

    __nv_fp8_e4m3* proj_mlp_weight;
    T*             proj_mlp_weight_h;
    T*             proj_mlp_bias;
    float*         proj_mlp_weight_scale;
    float*         proj_mlp_input_scale;

    __nv_fp8_e4m3* proj_out_weight;
    T*             proj_out_weight_h;
    T*             proj_out_bias;
    float*         proj_out_weight_scale;
    float*         proj_out_input_scale;

    FluxSingleTransformerFP8BlockWeight() = default;
    FluxSingleTransformerFP8BlockWeight(size_t        embedding_dim,
                                        size_t        embedding_head_num,
                                        size_t        embedding_head_dim,
                                        size_t        mlp_scale,
                                        LyraQuantType quant_level,
                                        IAllocator*   allocator);

    virtual ~FluxSingleTransformerFP8BlockWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
