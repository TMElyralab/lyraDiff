#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNormWeight.h"
#include "src/lyradiff/layers/flux_attention_processor/FluxAttentionProcessorWeight.h"
#include "src/lyradiff/layers/flux_attn_post_processor/FluxAttnPostProcessorWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class FluxTransformerBlockWeight: public IFLoraWeight<T> {
private:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;
    size_t mlp_scale_;

protected:
    bool is_maintain_buffer = false;

public:
    AdaLayerNormWeight<T>*           ada_norm_weight;
    AdaLayerNormWeight<T>*           context_ada_norm_weight;
    FluxAttentionProcessorWeight<T>* attn_weight;
    FluxAttnPostProcessorWeight<T>*  post_attn_weight;
    FluxAttnPostProcessorWeight<T>*  context_post_attn_weight;

    FluxTransformerBlockWeight() = default;
    FluxTransformerBlockWeight(size_t        embedding_dim,
                               size_t        embedding_head_num,
                               size_t        embedding_head_dim,
                               size_t        mlp_scale,
                               LyraQuantType quant_level,
                               IAllocator*   allocator);

    virtual ~FluxTransformerBlockWeight();

    virtual void loadWeights(const std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);

    virtual void mallocWeights();
};

}  // namespace lyradiff
