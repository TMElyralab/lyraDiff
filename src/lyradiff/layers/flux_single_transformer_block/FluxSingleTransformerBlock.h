#pragma once

#include "FluxSingleTransformerBlockWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaLayerNorm.h"
#include "src/lyradiff/layers/flux_single_attention_processor/FluxSingleAttentionProcessor.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxSingleTransformerBlock: public BaseLayer {
private:
    // params for 2 Identical ResNets

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

    T* norm_buffer;
    T* norm_buffer2;
    T* attn_output_buffer;
    T* msa_buffer;
    T* mlp_buffer1;
    T* mlp_buffer2;

    AdaLayerNorm<T>*                 ada_norm;
    FluxSingleAttentionProcessor<T>* attn_processor;

protected:
    size_t embedding_dim_;
    size_t embedding_head_num_;
    size_t embedding_head_dim_;
    size_t mlp_scale_;

public:
    FluxSingleTransformerBlock(size_t           embedding_dim,
                               size_t           embedding_head_num,
                               size_t           embedding_head_dim,
                               size_t           mlp_scale,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               const bool       is_free_buffer_after_forward,
                               const bool       sparse,
                               LyraQuantType    quant_level);

    FluxSingleTransformerBlock(FluxSingleTransformerBlock<T> const& other);

    virtual ~FluxSingleTransformerBlock();

    virtual void forward(const TensorMap*                           output_tensors,
                         const TensorMap*                           input_tensors,
                         const FluxSingleTransformerBlockWeight<T>* weights);
};

}  // namespace lyradiff
