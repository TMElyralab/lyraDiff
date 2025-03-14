#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

// clang-format off
template<typename T> struct GeluActivation;
// clang-format on

template<typename T>
void invokeaddBiasToGEMM(
    T* dst, const T* bias, const int batch_size, const int seq_len, const int dim, cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasGeglu(T*           dst,
                             const T*     hidden_states,
                             const T*     bias,
                             const int    seq_len,
                             const int    dim,
                             const int    batch_size,
                             cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasGegluV2(T*           dst,
                               const T*     hidden_states,
                               const T*     bias,
                               const int    seq_len,
                               const int    dim,
                               const int    batch_size,
                               cudaStream_t stream);
}  // namespace lyradiff
