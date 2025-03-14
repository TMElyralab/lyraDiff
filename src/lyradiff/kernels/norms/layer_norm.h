#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeLayerNorm(T*           dst,
                     const T*     src,
                     const T*     gamma,
                     const T*     beta,
                     size_t       batch_size,
                     size_t       channels,
                     size_t       nhiddens,
                     cudaStream_t stream,
                     const double eps = 1e-5);

template<typename T>
void invokeLayerNormWithShiftAndScale(T*           dst,
                                      const T*     src,
                                      const T*     scale,
                                      const T*     shift,
                                      size_t       batch_size,
                                      size_t       channels,
                                      size_t       nhiddens,
                                      cudaStream_t stream,
                                      const double eps = 1e-6);
}  // namespace lyradiff
