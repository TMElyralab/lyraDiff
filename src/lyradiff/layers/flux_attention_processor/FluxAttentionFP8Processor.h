#pragma once

#include "FluxAttentionFP8ProcessorWeight.h"
#include "FluxAttentionProcessor.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/attention/flash_fused_attention_v2/fmha.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxAttentionFP8Processor: public FluxAttentionProcessor<T> {
private:
    // params for 2 Identical ResNets
    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t encoder_seq_len);

    __nv_fp8_e4m3* fp8_buffer;

    T* qkv_buffer;
    T* encoder_qkv_buffer;

    T* nhidden_buffer;
    T* attn_out_buffer;

    // T* qkv_buffer;

    lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>* flash_attn_layer;

public:
    FluxAttentionFP8Processor(size_t           embedding_dim,
                              size_t           embedding_head_num,
                              size_t           embedding_head_dim,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse);

    FluxAttentionFP8Processor(FluxAttentionFP8Processor<T> const& other);

    virtual ~FluxAttentionFP8Processor();

    virtual void forward(const TensorMap*                          output_tensors,
                         const TensorMap*                          input_tensors,
                         const FluxAttentionFP8ProcessorWeight<T>* weights);
};

}  // namespace lyradiff
