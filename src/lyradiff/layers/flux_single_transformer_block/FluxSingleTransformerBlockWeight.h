#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNormWeight.h"
#include "src/lyradiff/layers/flux_single_attention_processor/FluxSingleAttentionProcessorWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxSingleTransformerBlockWeight: public IFLoraWeight<T> {
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

    T* proj_mlp_weight;
    T* proj_mlp_weight_h;
    T* proj_mlp_bias;

    T* proj_out_weight;
    T* proj_out_weight_h;
    T* proj_out_bias;

    FluxSingleTransformerBlockWeight() = default;
    FluxSingleTransformerBlockWeight(size_t        embedding_dim,
                                     size_t        embedding_head_num,
                                     size_t        embedding_head_dim,
                                     size_t        mlp_scale,
                                     LyraQuantType quant_level,
                                     IAllocator*   allocator);

    virtual ~FluxSingleTransformerBlockWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
