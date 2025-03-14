#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {
// split attn_output to encoder_hidden_state and nhidden
template<typename T>
void invokeSpiltEncoderAndHidden(T*           hidden_out,
                                 T*           encoder_hidden_out,
                                 const T*     attn_output,
                                 int          batch_size,
                                 int          hidden_seq_len,
                                 int          encoder_seq_len,
                                 int          dim,
                                 cudaStream_t stream);

template<typename T>
void invokeCatEncoderAndHidden(T*           output,
                               const T*     encoder_hidden,
                               const T*     hidden,
                               int          batch_size,
                               int          hidden_seq_len,
                               int          encoder_seq_len,
                               int          dim,
                               cudaStream_t stream);

}  // namespace lyradiff
