#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

namespace lyradiff {

template<typename T>
void invokeMulGammaAndAddBeta(T*           dst,
                              const T*     src,
                              const T*     gamma,
                              const T*     beta,
                              const size_t batch_size,
                              const size_t height,
                              const size_t width,
                              const size_t channels,
                              cudaStream_t stream = 0);

template<typename T>
void invokeMulControlScale(T*           dst,
                           const T*     src,
                           const T*     src2,
                           const size_t batch_size,
                           const size_t height,
                           const size_t width,
                           const size_t channels,
                           const float  control_scale,
                           cudaStream_t stream = 0);

}  // namespace lyradiff
