#pragma once

#include "FluxSingleAttentionFP8ProcessorWeight.h"
#include "FluxSingleAttentionProcessor.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/attention/flash_fused_attention_v2/fmha.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxSingleAttentionFP8Processor: public FluxSingleAttentionProcessor<T> {
private:
    // params for 2 Identical ResNets
    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

    __nv_fp8_e4m3* input_buffer;

    T* qkv_buffer1;
    T* qkv_buffer2;

    lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>* flash_attn_layer;

public:
    FluxSingleAttentionFP8Processor(size_t           embedding_dim,
                                    size_t           embedding_head_num,
                                    size_t           embedding_head_dim,
                                    cudaStream_t     stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator*      allocator,
                                    const bool       is_free_buffer_after_forward,
                                    const bool       sparse);

    FluxSingleAttentionFP8Processor(FluxSingleAttentionFP8Processor<T> const& other);

    virtual ~FluxSingleAttentionFP8Processor();

    virtual void forward(const TensorMap*                                output_tensors,
                         const TensorMap*                                input_tensors,
                         const FluxSingleAttentionFP8ProcessorWeight<T>* weights);
};

}  // namespace lyradiff
