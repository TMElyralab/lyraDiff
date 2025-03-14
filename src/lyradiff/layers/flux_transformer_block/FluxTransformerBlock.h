#pragma once

#include "FluxTransformerBlockWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNorm.h"
#include "src/lyradiff/layers/flux_attention_processor/FluxAttentionProcessor.h"
#include "src/lyradiff/layers/flux_attn_post_processor/FluxAttnPostProcessor.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxTransformerBlock: public BaseLayer {
private:
    // params for 2 Identical ResNets

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t context_seq_len);

    T* norm_buffer;
    T* context_norm_buffer;

    T* msa_buffer;
    T* context_msa_buffer;

    T* attn_output_buffer;
    T* context_attn_output_buffer;

    AdaLayerNorm<T>*           ada_norm;
    FluxAttentionProcessor<T>* attn_processor;
    FluxAttnPostProcessor<T>*  post_processor;

protected:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;
    size_t mlp_scale_;

public:
    FluxTransformerBlock(size_t              embedding_dim,
                         size_t              embedding_head_num,
                         size_t              embedding_head_dim,
                         size_t              mlp_scale,
                         cudaStream_t        stream,
                         cublasMMWrapper*    cublas_wrapper,
                         IAllocator*         allocator,
                         const bool          is_free_buffer_after_forward,
                         const bool          sparse,
                         const LyraQuantType quant_level);

    FluxTransformerBlock(FluxTransformerBlock<T> const& other);

    virtual ~FluxTransformerBlock();

    virtual void forward(const TensorMap*                     output_tensors,
                         const TensorMap*                     input_tensors,
                         const FluxTransformerBlockWeight<T>* weights);
};

}  // namespace lyradiff
