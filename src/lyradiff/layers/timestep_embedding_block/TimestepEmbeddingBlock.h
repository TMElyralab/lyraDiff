#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class TimestepEmbeddingBlock: public BaseLayer {
private:
    // params
    size_t input_dim_;
    size_t output_dim_0_;
    size_t output_dim_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size);

public:
    T* inter_buf_0_0_ = nullptr;  // for Linear0
    T* inter_buf_0_1_ = nullptr;  // for FusedAddBiasSiLu0

    T* inter_buf_1_0_ = nullptr;  // for Linear1
    T* inter_buf_1_1_ = nullptr;  // for AddBias1

    size_t inter_size_0_;
    size_t inter_size_1_;

    TimestepEmbeddingBlock(const size_t     input_dim_,
                           const size_t     output_dim_0_,
                           const size_t     output_dim_,
                           cudaStream_t     stream,
                           cublasMMWrapper* cublas_wrapper,
                           IAllocator*      allocator,
                           const bool       is_free_buffer_after_forward,
                           const bool       sparse);

    TimestepEmbeddingBlock(TimestepEmbeddingBlock<T> const& timestep_embedding_block);

    virtual ~TimestepEmbeddingBlock();

    virtual void forward(TensorMap*                             output_tensors,
                         const TensorMap*                       input_tensors,
                         const TimestepEmbeddingBlockWeight<T>* timestep_embedding_block_weights);

    virtual void forward(Tensor&                                output_tensor,
                         const Tensor&                          input_tensor,
                         const TimestepEmbeddingBlockWeight<T>* timestep_embedding_block_weights);
};

}  // namespace lyradiff
