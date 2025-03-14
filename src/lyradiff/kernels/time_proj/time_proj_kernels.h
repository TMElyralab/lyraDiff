#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeTimeProjection(T*           dst,
                          const float* timestep,
                          const float* exponents,
                          const bool&  flip_sin_to_cos_,
                          const size_t half_dim_,
                          const size_t batch_size,
                          cudaStream_t stream);
// if timestep is a vector
template<typename T>
void invokeTimeProjectionMulti(T*           dst,
                               const float* timestep,
                               int          n_timestep,
                               const float* exponents,
                               const bool&  flip_sin_to_cos_,
                               const size_t half_dim_,
                               const size_t batch_size,
                               cudaStream_t stream);

}  // namespace lyradiff
