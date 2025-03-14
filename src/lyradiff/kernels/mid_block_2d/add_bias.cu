#include "add_bias.h"
#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;

namespace lyradiff {
// Only Add bias
template<typename T>
__global__ void AddBias(T* dst, const T* hidden_states, const T* bias, const size_t dim, const size_t batch_size)
{
    // grid x 维度代表, batch_idx * seq_len
    // block x 维度为 dim / 4
    int dst_offset  = blockIdx.x * dim + threadIdx.x * 4;
    int bias_offset = threadIdx.x * 4;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

    T4 hidden_state_4      = *reinterpret_cast<const T4*>(hidden_states + dst_offset);
    T4 hidden_state_bias_4 = *reinterpret_cast<const T4*>(bias + bias_offset);

    hidden_state_4.x += hidden_state_bias_4.x;
    hidden_state_4.y += hidden_state_bias_4.y;
    hidden_state_4.z += hidden_state_bias_4.z;
    hidden_state_4.w += hidden_state_bias_4.w;

    *reinterpret_cast<T4*>(&dst[dst_offset]) = hidden_state_4;
}

template<typename T>
void invokeAddBias(T*           dst,
                   const T*     hidden_states,
                   const T*     bias,
                   const size_t seq_len,
                   const size_t dim,
                   const size_t batch_size,
                   cudaStream_t stream)
{
    // dim 的长度为 1280
    int threadsPerBlock = dim / 4;
    int numBlocks       = batch_size * seq_len;
    AddBias<<<numBlocks, threadsPerBlock, 0, stream>>>(dst, hidden_states, bias, dim, batch_size);
}

template<typename T>
__global__ void FusedAddBiasSplitQKV(T*           q_dst,
                                     T*           k_dst,
                                     T*           v_dst,
                                     const T*     hidden_states,
                                     const T*     bias,
                                     const size_t seq_len,
                                     const size_t dim_per_head,
                                     const size_t batch_size)
{
    // grid x 维度代表, batch_idx * seq_len
    // block x 维度为 dim_per_head * 3 / 4
    int input_offset = blockIdx.x * dim_per_head * 3 + threadIdx.x * 4;
    int dim_offset   = threadIdx.x * 4;
    int dst_offset   = blockIdx.x * dim_per_head + dim_offset % dim_per_head;

    using T4 = typename TypeConverter4<T>::Type;  // float to float4, half to half4

    T4 hidden_state_4      = *reinterpret_cast<const T4*>(hidden_states + input_offset);
    T4 hidden_state_bias_4 = *reinterpret_cast<const T4*>(bias + dim_offset);

    hidden_state_4.x += hidden_state_bias_4.x;
    hidden_state_4.y += hidden_state_bias_4.y;
    hidden_state_4.z += hidden_state_bias_4.z;
    hidden_state_4.w += hidden_state_bias_4.w;

    if (dim_offset / dim_per_head == 0) {
        *reinterpret_cast<T4*>(&q_dst[dst_offset]) = hidden_state_4;
    }
    else if (dim_offset / dim_per_head == 1) {
        *reinterpret_cast<T4*>(&k_dst[dst_offset]) = hidden_state_4;
    }
    else {
        *reinterpret_cast<T4*>(&v_dst[dst_offset]) = hidden_state_4;
    }
}

// Add bias and split hidden_states to qkv, since the head num here is 1, we don't need to do the transpose
// so input -> outputs are [bs, seq_len, 3, dim per head] -> [bs, seq_len, dim per head] * 3

template<typename T>
void invokeFusedAddBiasSplitQKV(T*           q_dst,
                                T*           k_dst,
                                T*           v_dst,
                                const T*     hidden_states,
                                const T*     bias,
                                const size_t seq_len,
                                const size_t dim_per_head,
                                const size_t batch_size,
                                cudaStream_t stream)
{
    int threadsPerBlock = dim_per_head * 3 / 4;
    int numBlocks       = batch_size * seq_len;
    FusedAddBiasSplitQKV<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        q_dst, k_dst, v_dst, hidden_states, bias, seq_len, dim_per_head, batch_size);
}

#define INSTANTIATE_INVOKE_ADD_BIAS(T)                                                                                 \
    template void invokeAddBias(T*           dst,                                                                      \
                                const T*     hidden_states,                                                            \
                                const T*     bias,                                                                     \
                                const size_t seq_len,                                                                  \
                                const size_t dim,                                                                      \
                                const size_t batch_size,                                                               \
                                cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_BIAS(float);
INSTANTIATE_INVOKE_ADD_BIAS(half);
#undef INSTANTIATE_INVOKE_ADD_BIAS

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SPLIT_QKV(T)                                                                 \
    template void invokeFusedAddBiasSplitQKV(T*           q_dst,                                                       \
                                             T*           k_dst,                                                       \
                                             T*           v_dst,                                                       \
                                             const T*     hidden_states,                                               \
                                             const T*     bias,                                                        \
                                             const size_t seq_len,                                                     \
                                             const size_t dim_per_head,                                                \
                                             const size_t batch_size,                                                  \
                                             cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SPLIT_QKV(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SPLIT_QKV(half);
#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SPLIT_QKV
}  // namespace lyradiff