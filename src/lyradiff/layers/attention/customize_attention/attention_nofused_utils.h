#pragma once
#include "src/lyradiff/common.h"

namespace lyradiff {
template<typename T>
__global__ void add_QKV_bias(const T*   QKV,
                             const T*   bias_QKV,
                             T*         q_buf,
                             T*         k_buf,
                             T*         v_buf,
                             const int  batch_size,
                             const int  seq_len,
                             const int  head_num,
                             const int  half_size_per_head,
                             const bool is_roformer);

template<typename T>
__global__ void
transpose(const T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head);

}  // namespace lyradiff