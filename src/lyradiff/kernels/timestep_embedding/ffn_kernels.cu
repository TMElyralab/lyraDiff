#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

using namespace std;

namespace lyradiff {
// Only Add bias
template<typename T>
__global__ void AddBias1D(T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size)
{
    // grid x 维度代表, batch_idx
    // block x 维度为 dim / 2
    int dst_offset  = blockIdx.x * dim + threadIdx.x * 2;
    int bias_offset = threadIdx.x * 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    T2 hidden_state_2      = *reinterpret_cast<const T2*>(hidden_states + dst_offset);
    T2 hidden_state_bias_2 = *reinterpret_cast<const T2*>(bias + bias_offset);

    hidden_state_2.x += hidden_state_bias_2.x;
    hidden_state_2.y += hidden_state_bias_2.y;

    *reinterpret_cast<T2*>(&dst[dst_offset]) = hidden_state_2;
}

template<typename T>
void invokeAddBias1D(
    T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream)
{
    // dim 的长度为 1280
    // if dim=5120, threadsPerBlock
    int threadsPerBlock = dim / 2;
    int numBlocks       = batch_size;
    AddBias1D<<<numBlocks, threadsPerBlock, 0, stream>>>(dst, hidden_states, bias, dim, batch_size);
}

#define INSTANTIATE_INVOKE_ADD_BIAS_1D(T)                                                                              \
    template void invokeAddBias1D(                                                                                     \
        T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_BIAS_1D(float);
INSTANTIATE_INVOKE_ADD_BIAS_1D(half);
#undef INSTANTIATE_INVOKE_ADD_BIAS_1D

// Fused Add bias with Silu activation
template<typename T>
__global__ void fusedAddBiasSilu1D(T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size)
{
    // grid x 维度代表, batch_idx
    // block x 维度为 dim / 2
    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        int dst_offset  = blockIdx.x * dim + idx * 2;
        int bias_offset = idx * 2;

        using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

        T2 hidden_state_2      = *reinterpret_cast<const T2*>(hidden_states + dst_offset);
        T2 hidden_state_bias_2 = *reinterpret_cast<const T2*>(bias + bias_offset);

        hidden_state_2.x += hidden_state_bias_2.x;
        hidden_state_2.y += hidden_state_bias_2.y;

        // float2 hidden_state_2_f      = cuda_cast<float2>(hidden_state_2);
        // float2 hidden_state_bias_2_f = cuda_cast<float2>(hidden_state_bias_2);

        // hidden_state_2_f.x += hidden_state_bias_2_f.x;
        // hidden_state_2_f.y += hidden_state_bias_2_f.y;

        hidden_state_2.x = SiluActivation<T>::apply(hidden_state_2.x);
        hidden_state_2.y = SiluActivation<T>::apply(hidden_state_2.y);

        *reinterpret_cast<T2*>(&dst[dst_offset]) = hidden_state_2;
    }
}

template<typename T>
void invokeFusedAddBiasSilu1D(
    T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream)
{
    // dim 的长度为 1280
    // int threadsPerBlock = dim / 2;
    dim3 block(std::min(dim / 2, 1024));

    int numBlocks = batch_size;
    fusedAddBiasSilu1D<<<numBlocks, block, 0, stream>>>(dst, hidden_states, bias, dim, batch_size);
}

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SILU_1D(T)                                                                   \
    template void invokeFusedAddBiasSilu1D(                                                                            \
        T* dst, const T* hidden_states, const T* bias, const int dim, const int batch_size, cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SILU_1D(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SILU_1D(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SILU_1D(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_SILU_1D

}  // namespace lyradiff