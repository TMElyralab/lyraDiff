#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {
// fused_cat_attn_out_and_act_mlp
template<typename T>
void invokeFusedCatAndGelu(T*           out,
                           const T*     attn_output,
                           const T*     mlp_output,
                           int          batch_size,
                           int          seq_len,
                           int          attn_dim,
                           int          mlp_dim,
                           cudaStream_t stream);

// fused_gate_and_add_residual
template<typename T>
void invokeFusedGateAndResidual(T*           out,
                                const T*     input,
                                const T*     gate,
                                const T*     residual,
                                int          batch_size,
                                int          seq_len,
                                int          dim,
                                cudaStream_t stream);

template<typename T>
void invokeFusedBiasAndResidual(T*           out,
                                const T*     input,
                                const T*     bias,
                                const T*     residual,
                                int          batch_size,
                                int          seq_len,
                                int          dim,
                                cudaStream_t stream);

template<typename T>
void invokeAddBias(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream);

template<typename T>
void invokeFusedBiasAndGelu(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream);

}  // namespace lyradiff
