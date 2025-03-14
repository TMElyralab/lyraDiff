#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeAddBias(T*           dst,
                   const T*     hidden_states,
                   const T*     bias,
                   const size_t seq_len,
                   const size_t dim,
                   const size_t batch_size,
                   cudaStream_t stream);

template<typename T>
void invokeFusedAddBiasSplitQKV(T*           q_dst,
                                T*           k_dst,
                                T*           v_dst,
                                const T*     hidden_states,
                                const T*     bias,
                                const size_t seq_len,
                                const size_t dim_per_head,
                                const size_t batch_size,
                                cudaStream_t stream);

}  // namespace lyradiff
