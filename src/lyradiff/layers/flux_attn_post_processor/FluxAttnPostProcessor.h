#pragma once

#include "FluxAttnPostProcessorWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class FluxAttnPostProcessor: public BaseLayer {
private:
    // params for 2 Identical ResNets

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

    T* hidden_buffer;
    T* hidden_buffer2;

    T* ffn_inner_buffer;

protected:
    size_t embedding_dim_;

public:
    FluxAttnPostProcessor(size_t           embedding_dim,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          const bool       is_free_buffer_after_forward,
                          const bool       sparse);

    FluxAttnPostProcessor(FluxAttnPostProcessor<T> const& other);

    virtual ~FluxAttnPostProcessor();

    virtual void forward(const TensorMap*                      output_tensors,
                         const TensorMap*                      input_tensors,
                         const FluxAttnPostProcessorWeight<T>* weights);
};

}  // namespace lyradiff
