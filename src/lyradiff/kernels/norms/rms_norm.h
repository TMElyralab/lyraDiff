#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeRootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream);

}  // namespace lyradiff
