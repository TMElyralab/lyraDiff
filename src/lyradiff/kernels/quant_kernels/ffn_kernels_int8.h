#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

// clang-format off
template<typename T> struct GeluActivation;
// clang-format on

template<typename T>
void invokeFusedAddBiasGegluInt8O(int8_t*      dst,
                                  const T*     hidden_states,
                                  const T*     bias,
                                  const float* pre_quant_scale,
                                  const float* input_quant_scale,
                                  const int    seq_len,
                                  const int    dim,
                                  const int    batch_size,
                                  cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasGegluInt8OV2(int8_t*      dst,
                                    const T*     hidden_states,
                                    const T*     bias,
                                    const float* pre_quant_scale,
                                    const float* input_quant_scale,
                                    const int    seq_len,
                                    const int    dim,
                                    const int    batch_size,
                                    cudaStream_t stream);

}  // namespace lyradiff
