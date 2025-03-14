#pragma once
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/utils/quantization.h"
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeLayerNormInt8O(int8_t*      dst,
                          const T*     src,
                          const T*     gamma,
                          const T*     beta,
                          const float* pre_quant_scale,
                          const float* input_quant_scale,
                          QuantMode    quant_mode,
                          size_t       batch_size,
                          size_t       channels,
                          size_t       nhiddens,
                          cudaStream_t stream,
                          const double eps = 1e-6);

template<typename T>
void invokeFusedResidualBiasLayerNormInt8O(int8_t*      dst,  // layernorm 以及
                                           T*           dst2, // fused add bias and residual 之后的结果
                                           const T*     src,
                                           const T*     residual,
                                           const T*     bias,
                                           const T*     gamma,
                                           const T*     beta,
                                           const float* pre_quant_scale,
                                           const float* input_quant_scale,
                                           QuantMode    quant_mode,
                                           size_t       batch_size,
                                           size_t       channels,
                                           size_t       nhiddens,
                                           cudaStream_t stream,
                                           const double eps = 1e-6);

}  // namespace lyradiff
