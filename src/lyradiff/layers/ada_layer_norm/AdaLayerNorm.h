#pragma once

#include "AdaLayerNormWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class AdaLayerNorm: public BaseLayer {
private:
    // params for 2 Identical ResNets
    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size);

    T* msa_buffer;

protected:
    size_t embedding_dim_;
    size_t embedding_scale_;
    bool   switch_scale_;

public:
    AdaLayerNorm(size_t           embedding_dim,
                 size_t           embedding_scale,
                 bool             switch_scale,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 const bool       is_free_buffer_after_forward,
                 const bool       sparse);

    AdaLayerNorm(AdaLayerNorm<T> const& other);

    virtual ~AdaLayerNorm();

    virtual void
    forward(const TensorMap* output_tensors, const TensorMap* input_tensors, const AdaLayerNormWeight<T>* weights);
};

}  // namespace lyradiff
