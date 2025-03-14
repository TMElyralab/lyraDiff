#include "attention_nofused_utils.h"

namespace lyradiff {
template<>
__global__ void add_QKV_bias<float>(const float* QKV,
                                    const float* bias_QKV,
                                    float*       q_buf,
                                    float*       k_buf,
                                    float*       v_buf,
                                    const int    batch_size,
                                    const int    seq_len,
                                    const int    head_num,
                                    const int    half_size_per_head,
                                    const bool   is_roformer)
{
    int batch_id = blockIdx.y;
    int seq_id   = blockIdx.x;
    int head_id  = threadIdx.x / half_size_per_head;
    int id       = threadIdx.x % half_size_per_head;
    int src_id   = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
    int trt_id   = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;

    float2 q_value = ((float2*)QKV)[src_id], q_bias = __ldg(&((float2*)bias_QKV)[threadIdx.x]);
    float2 k_value = ((float2*)QKV)[src_id + blockDim.x],
           k_bias  = __ldg(&((float2*)bias_QKV)[threadIdx.x + blockDim.x]);
    float2 v_value = ((float2*)QKV)[src_id + blockDim.x * 2],
           v_bias  = __ldg(&((float2*)bias_QKV)[threadIdx.x + blockDim.x * 2]);
    q_value.x += q_bias.x, q_value.y += q_bias.y;
    k_value.x += k_bias.x, k_value.y += k_bias.y;
    v_value.x += v_bias.x, v_value.y += v_bias.y;

    if (is_roformer) {
        float2 ro_q         = make_float2(-q_value.y, q_value.x);
        float2 ro_k         = make_float2(-k_value.y, k_value.x);
        float  position_enc = __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
        float  sin_pos      = __sinf(position_enc);
        float  cos_pos      = __cosf(position_enc);
        q_value.x = q_value.x * cos_pos + ro_q.x * sin_pos, q_value.y = q_value.y * cos_pos + ro_q.y * sin_pos;
        k_value.x = k_value.x * cos_pos + ro_k.x * sin_pos, k_value.y = k_value.y * cos_pos + ro_k.y * sin_pos;
    }

    ((float2*)q_buf)[trt_id] = q_value;
    ((float2*)k_buf)[trt_id] = k_value;
    ((float2*)v_buf)[trt_id] = v_value;
}

template<>
__global__ void add_QKV_bias<__half>(const __half* QKV,
                                     const __half* bias_QKV,
                                     __half*       q_buf,
                                     __half*       k_buf,
                                     __half*       v_buf,
                                     const int     batch_size,
                                     const int     seq_len,
                                     const int     head_num,
                                     const int     half_size_per_head,
                                     const bool    is_roformer)
{
    int   batch_id = blockIdx.y;
    int   seq_id   = blockIdx.x;
    int   head_id  = threadIdx.x / half_size_per_head;
    int   id       = threadIdx.x % half_size_per_head;
    int   src_id   = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
    int   trt_id   = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
    half2 q_value  = __hadd2(((const half2*)QKV)[src_id], __ldg(&((const half2*)bias_QKV)[threadIdx.x]));
    half2 k_value =
        __hadd2(((const half2*)QKV)[src_id + blockDim.x], __ldg(&((const half2*)bias_QKV)[threadIdx.x + blockDim.x]));
    half2 v_value = __hadd2(((const half2*)QKV)[src_id + blockDim.x * 2],
                            __ldg(&((const half2*)bias_QKV)[threadIdx.x + blockDim.x * 2]));

    if (is_roformer) {
        half2 ro_q         = half2(-q_value.y, q_value.x);
        half2 ro_k         = half2(-k_value.y, k_value.x);
        float position_enc = __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
        half2 sin_pos      = __float2half2_rn(__sinf(position_enc));
        half2 cos_pos      = __float2half2_rn(__cosf(position_enc));
        q_value            = __hadd2(__hmul2(q_value, cos_pos), __hmul2(ro_q, sin_pos));
        k_value            = __hadd2(__hmul2(k_value, cos_pos), __hmul2(ro_k, sin_pos));
    }

    ((half2*)q_buf)[trt_id] = q_value;
    ((half2*)k_buf)[trt_id] = k_value;
    ((half2*)v_buf)[trt_id] = v_value;
}

template<>
__global__ void transpose<float>(
    const float* src, float* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id    = blockIdx.x / seq_len;
    int seq_id      = blockIdx.x % seq_len;
    int head_id     = threadIdx.y;
    int src_offset  = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
    int dst_offset  = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
    dst[dst_offset] = src[src_offset];
}

template<>
__global__ void transpose<__half>(const __half* src,
                                  __half*       dst,
                                  const int     batch_size,
                                  const int     seq_len,
                                  const int     head_num,
                                  const int     size_per_head)
{
    int batch_id              = blockIdx.x / seq_len;
    int seq_id                = blockIdx.x % seq_len;
    int head_id               = threadIdx.y;
    int src_offset            = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
    int dst_offset            = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
    ((half2*)dst)[dst_offset] = ((const half2*)src)[src_offset];
}
}  // namespace lyradiff