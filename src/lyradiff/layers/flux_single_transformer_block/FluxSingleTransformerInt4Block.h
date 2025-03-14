#pragma once

#include "FluxSingleTransformerBlock.h"
#include "FluxSingleTransformerInt4BlockWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/ada_layer_norm/AdaFP8LayerNorm.h"
#include "src/lyradiff/layers/flux_single_attention_processor/FluxSingleAttentionInt4Processor.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxSingleTransformerInt4Block: public FluxSingleTransformerBlock<T> {
private:
    // params for 2 Identical ResNets
    // size_t embedding_dim_;
    // size_t embedding_head_num_;
    // size_t embedding_head_dim_;
    // size_t mlp_scale_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

    T* norm_buffer;
    T* norm_buffer2;
    T* attn_output_buffer;
    T* msa_buffer;
    T* mlp_buffer1;
    T* hidden_buffer1;
    T* hidden_buffer2;

    // __nv_fp8_e4m3* fp8_buffer1;
    // __nv_fp8_e4m3* fp8_buffer2;

    AdaLayerNorm<T>*                 ada_norm;
    FluxSingleAttentionProcessor<T>* attn_processor;

    W4A4Gemm<T>* proj_mlp_gemm;
    W4A4Gemm<T>* proj_out_gemm_1;
    W4A4Gemm<T>* proj_out_gemm_2;

public:
    FluxSingleTransformerInt4Block(size_t           embedding_dim,
                                   size_t           embedding_head_num,
                                   size_t           embedding_head_dim,
                                   size_t           mlp_scale,
                                   cudaStream_t     stream,
                                   cublasMMWrapper* cublas_wrapper,
                                   IAllocator*      allocator,
                                   const bool       is_free_buffer_after_forward,
                                   const bool       sparse,
                                   LyraQuantType    quant_level);

    FluxSingleTransformerInt4Block(FluxSingleTransformerInt4Block<T> const& other);

    virtual ~FluxSingleTransformerInt4Block();

    virtual void forward(const TensorMap*                               output_tensors,
                         const TensorMap*                               input_tensors,
                         const FluxSingleTransformerInt4BlockWeight<T>* weights);
};

}  // namespace lyradiff
