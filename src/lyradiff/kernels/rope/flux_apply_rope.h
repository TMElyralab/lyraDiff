#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

// apply rope for BSND
template<typename T>
void invokeFluxApplyRope(T*           out,
                         const T*     input,
                         const T*     rope_emb,
                         int          batch_size,
                         int          seq_len,
                         int          head_num,
                         int          head_dim,
                         cudaStream_t stream);

template<typename T>
void invokeFusedRmsNormAndRope(T*           out,
                               const T*     input,
                               const T*     scale,
                               const T*     rope_emb,
                               float        eps,
                               int          batch_size,
                               int          seq_len,
                               int          head_num,
                               int          head_dim,
                               cudaStream_t stream);

template<typename T>
void invokeFusedBiasRmsNormAndRope(T*           out,
                                   const T*     input,
                                   const T*     bias,
                                   const T*     scale,
                                   const T*     rope_emb,
                                   float        eps,
                                   int          batch_size,
                                   int          seq_len,
                                   int          head_num,
                                   int          head_dim,
                                   cudaStream_t stream);

template<typename T>
void invokeFusedRmsNormCatAndRope(T*           out,
                                  const T*     hidden,
                                  const T*     encoder_hidden,
                                  const T*     hidden_rms_scale,
                                  const T*     encoder_rms_scale,
                                  const T*     rope_emb,
                                  float        eps,
                                  int          batch_size,
                                  int          hidden_seq_len,
                                  int          encoder_seq_len,
                                  int          head_num,
                                  int          head_dim,
                                  cudaStream_t stream);

}  // namespace lyradiff
