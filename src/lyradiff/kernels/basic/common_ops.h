#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeConcat2d(T* dst, const T* src1, const T* src2, int batch_size, int dim1, int dim2, cudaStream_t stream);

template<typename T>
void invokeAddTensor2d(T* dst, const T* src1, const T* src2, size_t batch_size, size_t dim, cudaStream_t stream);

template<typename T>
void invokeAdd3Tensor2d(
    T* dst, const T* src1, const T* src2, const T* src3, size_t batch_size, size_t dim, bool silu, cudaStream_t stream);

template<typename T>
void invokeTranspose102(T* dst, T* src, const int dim0, const int dim1, const int dim2);
}  // namespace lyradiff
