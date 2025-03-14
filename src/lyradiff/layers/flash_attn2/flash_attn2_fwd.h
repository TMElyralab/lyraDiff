#pragma once

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// template<typename T>
// void set_params_fprop(Flash_fwd_params& params,
//                       // sizes
//                       const size_t b,
//                       const size_t seqlen_q,
//                       const size_t seqlen_k,
//                       const size_t seqlen_q_rounded,
//                       const size_t seqlen_k_rounded,
//                       const size_t h,
//                       const size_t h_k,
//                       const size_t d,
//                       const size_t d_rounded,
//                       // device pointers
//                       const T*                     q,
//                       const T*                     k,
//                       const T*                     v,
//                       T*                           out,
//                       const std::vector<uint32_t>& q_stride,
//                       const std::vector<uint32_t>& k_stride,
//                       const std::vector<uint32_t>& v_stride,
//                       const std::vector<uint32_t>& out_stride,
//                       void*                        cu_seqlens_q_d,
//                       void*                        cu_seqlens_k_d,
//                       void*                        p_d,
//                       void*                        softmax_lse_d,
//                       float                        p_dropout,
//                       float                        softmax_scale,
//                       bool                         is_causal);

template<typename T>
void invokeFlashAttn2Fwd(T*                           out,          // batch_size x seqlen_q x num_heads x head_size
                         float*                       softmax_lse,  // batch_size x num_heads x seqlen_q  float32
                         const T*                     q,            // batch_size x seqlen_q x num_heads x head_size
                         const T*                     k,            // batch_size x seqlen_k x num_heads_k x head_size
                         const T*                     v,            // batch_size x seqlen_k x num_heads_k x head_size
                         const std::vector<uint32_t>& q_shape,
                         const std::vector<uint32_t>& k_shape,
                         const std::vector<uint32_t>& v_shape,
                         const std::vector<uint32_t>& out_shape,
                         const float                  softmax_scale,
                         const bool                   is_causal,
                         cudaStream_t                 stream);
