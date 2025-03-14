#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeAddResidual(T*           dst,
                       const T*     src1,
                       const T*     src2,
                       const float  scale,
                       const size_t channel,
                       const size_t height,
                       const size_t width,
                       const int    batch_size,
                       cudaStream_t stream);

template<typename T>
void invokeAddResidualWithDifferentBatch(T*           dst,
                                         const T*     src1,
                                         const T*     src2,
                                         const float  scale,
                                         const size_t channel,
                                         const size_t height,
                                         const size_t width,
                                         const int    batch_size,
                                         const int    batch_size_2,
                                         cudaStream_t stream);

}  // namespace lyradiff
