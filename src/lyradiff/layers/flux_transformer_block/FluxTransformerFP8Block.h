#pragma once

#include "FluxTransformerBlock.h"
#include "FluxTransformerFP8BlockWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaFP8LayerNorm.h"
#include "src/lyradiff/layers/flux_attention_processor/FluxAttentionFP8Processor.h"
#include "src/lyradiff/layers/flux_attn_post_processor/FluxAttnPostFP8Processor.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxTransformerFP8Block: public FluxTransformerBlock<T> {
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

    AdaLayerNorm<T>*              ada_norm;
    FluxAttentionFP8Processor<T>* attn_processor;
    FluxAttnPostFP8Processor<T>*  post_processor;

public:
    FluxTransformerFP8Block(size_t              embedding_dim,
                            size_t              embedding_head_num,
                            size_t              embedding_head_dim,
                            size_t              mlp_scale,
                            cudaStream_t        stream,
                            cublasMMWrapper*    cublas_wrapper,
                            IAllocator*         allocator,
                            const bool          is_free_buffer_after_forward,
                            const bool          sparse,
                            const LyraQuantType quant_level);

    FluxTransformerFP8Block(FluxTransformerFP8Block<T> const& other);

    virtual ~FluxTransformerFP8Block();

    virtual void forward(const TensorMap*                        output_tensors,
                         const TensorMap*                        input_tensors,
                         const FluxTransformerFP8BlockWeight<T>* weights);
};

}  // namespace lyradiff
