#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

namespace lyradiff {
template<typename T, typename T_IN>
struct MaskedSoftmaxParam {
    // Common parameters.
    T*          attention_score = nullptr;  // (batch_size, head_num, q_length, k_length)
    const T_IN* qk              = nullptr;  // (batch_size, head_num, q_length, k_length)
    const T*    attention_mask  = nullptr;  // (batch_size, q_length, k_length)
    int         batch_size      = 0;
    int         q_length        = 0;
    int         k_length        = 0;
    int         num_heads       = 0;
    T           qk_scale        = T(0.0f);

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    const T* linear_bias_slopes = nullptr;  // (head_num,), optional
};

template<typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template<typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype* dst, const Datatype* src);

template<>
__device__ __inline__ void copy_vector<half, 1>(half* dst, const half* src)
{
    *dst = *src;
}

template<>
__device__ __inline__ void copy_vector<half, 4>(half* dst, const half* src)
{
    *((float2*)dst) = *((float2*)src);
}

template<>
__device__ __inline__ void copy_vector<float, 1>(float* dst, const float* src)
{
    *dst = *src;
}

template<>
__device__ __inline__ void copy_vector<float, 4>(float* dst, const float* src)
{
    *((float2*)dst) = *((float2*)src);
}

template<>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t* dst, const uint8_t* src)
{
    *dst = *src;
}

template<>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t* dst, const uint8_t* src)
{
    *((half2*)dst) = *((half2*)src);
}

template<typename input_t, typename output_t>
void dispatch_scaled_softmax_forward(output_t*      dst,
                                     const input_t* src,
                                     const input_t  scale,
                                     int            query_seq_len,
                                     int            key_seq_len,
                                     int            batches,
                                     int            attn_heads,
                                     cudaStream_t   stream);

}  // namespace lyradiff