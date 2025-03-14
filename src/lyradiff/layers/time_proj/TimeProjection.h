#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class TimeProjection: public BaseLayer {
private:
    // params
    size_t num_channels_;
    size_t half_dim_;
    bool   flip_sin_to_cos_;
    size_t downscale_freq_shift_;

    size_t max_period_;
    float  exponent_factor_;
    float* exponents_       = nullptr;
    float* timestep_buffer_ = nullptr;

    void allocateBuffer() override;
    void freeBuffer() override;

public:
    TimeProjection(const size_t     num_channels,
                   const bool       flip_sin_to_cos,
                   const size_t     downscale_freq_shift,
                   const size_t     max_period,
                   cudaStream_t     stream,
                   cublasMMWrapper* cublas_wrapper,
                   IAllocator*      allocator,
                   const bool       is_free_buffer_after_forward,
                   const bool       sparse);

    TimeProjection(const size_t     num_channels,
                   const bool       flip_sin_to_cos,
                   const size_t     downscale_freq_shift,
                   cudaStream_t     stream,
                   cublasMMWrapper* cublas_wrapper,
                   IAllocator*      allocator,
                   const bool       is_free_buffer_after_forward,
                   const bool       sparse);

    TimeProjection(TimeProjection<T> const& TimeProjection);

    virtual ~TimeProjection();

    virtual void forward(TensorMap* output_tensors, const float& timestep);
    virtual void forward(Tensor& output_tensor, const float& timestep);
    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors);
    virtual void forward(Tensor& output_tensors, Tensor& input_tensors);
};

}  // namespace lyradiff
