#pragma once

#include "FluxAttnPostInt4ProcessorWeight.h"
#include "FluxAttnPostProcessor.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/w4a4_gemm_block/W4A4Gemm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxAttnPostInt4Processor: public FluxAttnPostProcessor<T> {
private:
    // params for 2 Identical ResNets

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

    W4A4Gemm<T>* geglu_gemm;

    __nv_fp8_e4m3* fp8_buffer;

    T* hidden_buffer;
    T* hidden_buffer2;

    T* ffn_inner_buffer;

public:
    FluxAttnPostInt4Processor(size_t           embedding_dim,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse);

    FluxAttnPostInt4Processor(FluxAttnPostInt4Processor<T> const& other);

    virtual ~FluxAttnPostInt4Processor();

    virtual void forward(const TensorMap*                          output_tensors,
                         const TensorMap*                          input_tensors,
                         const FluxAttnPostInt4ProcessorWeight<T>* weights);
};

}  // namespace lyradiff
