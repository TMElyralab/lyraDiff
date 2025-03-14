#include "src/lyradiff/kernels/activation_kernels.h"

namespace lyradiff {

template<template<typename T> class Activation, typename T>
__global__ void generic_activation(T* dst, const T* src, const size_t length)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        dst[id] = cuda_cast<T>(Activation<T>::apply(src[id]));
    }
    // dst[id] = cuda_cast<T>(Activation<T>::apply(src[id]));
}
// clang-format on

template<template<typename T> class Activation, typename T>
void invokeGenericActivation(T* dst, const T* src, const size_t length, cudaStream_t stream)
{

    dim3 block, grid;
    if (length > 1024) {
        block.x = 1024;
        grid.x  = ceil(length / 1024.);
    }
    else {
        grid.x  = 1;
        block.x = length;
    }
    generic_activation<Activation><<<grid, block, 0, stream>>>(dst, src, length);
}

template void
invokeGenericActivation<SiluActivation, half>(half* dst, const half* src, const size_t length, cudaStream_t stream);
template void
invokeGenericActivation<SiluActivation, float>(float* dst, const float* src, const size_t length, cudaStream_t stream);

}  // namespace lyradiff