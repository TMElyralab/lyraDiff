#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeInterpolateNearest(T*           dst,
                              const T*     src,
                              size_t       bs,
                              size_t       height,
                              size_t       width,
                              size_t       channels,
                              size_t       up_factor,
                              cudaStream_t stream);

template<typename T>
void invokeInterpolateNearestToShape(T*           dst,
                                     const T*     src,
                                     size_t       bs,
                                     size_t       height,
                                     size_t       width,
                                     size_t       channels,
                                     size_t       tgt_height,
                                     size_t       tgt_width,
                                     cudaStream_t stream);
}  // namespace lyradiff
