#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

// clang-format off
template<typename T> struct SiluActivation;
// clang-format on

template<typename T>
void invokeAddBias1D(
    T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasSilu1D(
    T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream);

}  // namespace lyradiff
