#pragma once

#include "AdaFP8LayerNormWeight.h"
#include "AdaLayerNorm.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class AdaFP8LayerNorm: public AdaLayerNorm<T> {
private:
    // params for 2 Identical ResNets

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t input_size);

    __nv_fp8_e4m3* input_buffer;
    T*             msa_buffer;

public:
    AdaFP8LayerNorm(size_t           embedding_dim,
                    size_t           embedding_scale,
                    bool             switch_scale,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    const bool       is_free_buffer_after_forward,
                    const bool       sparse);

    AdaFP8LayerNorm(AdaFP8LayerNorm<T> const& other);

    virtual ~AdaFP8LayerNorm();

    virtual void
    forward(const TensorMap* output_tensors, const TensorMap* input_tensors, const AdaFP8LayerNormWeight<T>* weights);
};

}  // namespace lyradiff
