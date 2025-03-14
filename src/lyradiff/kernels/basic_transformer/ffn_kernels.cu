#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

using namespace std;

namespace lyradiff {

template<typename T>
__global__ void
addBiasToGEMM(T* dst, const T* bias){

    // grid layout: <<<batch_size, seq_len>>
    // threads per block: dim/32, each thread loop 32
    size_t bias_id = blockIdx.x*gridDim.y + blockIdx.y;
    int tid = threadIdx.x;
    int n_thread = blockDim.x;
#pragma unroll
    for(int i=0;i<32;i++){
        dst[bias_id*n_thread*32 + tid*32 + i] += bias[tid * 32 + i];
    }
}

template<typename T>
void invokeaddBiasToGEMM(T*           dst,
                         const T*     bias,
                         const int    batch_size,
                         const int    seq_len,
                         const int    dim,
                         cudaStream_t stream)
{
    // dim 的长度仅为 1280, 2560, 5120 的一种
    dim3 grid(batch_size, seq_len);
    int thread_per_block = dim / 32;
    // printf("dim: %d, thread_per_dim:%d\n", dim, thread_per_block);
    addBiasToGEMM<T><<<grid, thread_per_block, 0, stream>>>(dst, bias);
}


#define INSTANTIATE_INVOKE_ADD_BIAS_TO_GEMM(T)                                                                  \
    template void invokeaddBiasToGEMM(T*           dst,                                                          \
                                      const T*     bias,                                                         \
                                      const int    batch_size,                                                      \
                                      const int    seq_len,                                                          \
                                      const int    dim,                                                   \
                                      cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_BIAS_TO_GEMM(float);
INSTANTIATE_INVOKE_ADD_BIAS_TO_GEMM(half);
#undef INSTANTIATE_INVOKE_ADD_BIAS_TO_GEMM


template<typename T, int32_t DIM_PER_THREAD>
__global__ void
fusedAddBiasGeglu(T* dst, const T* hidden_states, const T* bias, const int seq_len, const int dim, const int batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2
    int batch_idx = blockIdx.y;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col             = col + i * 2;
        T2  hidden_state_2      = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col);
        T2  hidden_state_bias_2 = *reinterpret_cast<const T2*>(bias + cur_col);
        T2  gate_2      = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col + inner_col);
        T2  gate_bias_2 = *reinterpret_cast<const T2*>(bias + cur_col + inner_col);

        hidden_state_2.x += hidden_state_bias_2.x;
        hidden_state_2.y += hidden_state_bias_2.y;
        gate_2.x += gate_bias_2.x;
        gate_2.y += gate_bias_2.y;
        gate_2 = GeluActivation<T2>::apply(gate_2);

        dst[cur_row * inner_col + cur_col]     = hidden_state_2.x * gate_2.x;
        dst[cur_row * inner_col + cur_col + 1] = hidden_state_2.y * gate_2.y;
    }
}

template<typename T, int32_t DIM_PER_THREAD>
__global__ void
fusedGeglu(T* dst, const T* hidden_states, const int seq_len, const int dim, const int batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2
    int batch_idx = blockIdx.y;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col             = col + i * 2;
        T2  hidden_state_2      = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col);
        T2  gate_2      = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col + inner_col);

        //hidden_state_2.x += hidden_state_bias_2.x;
        //hidden_state_2.y += hidden_state_bias_2.y;
        //gate_2.x += gate_bias_2.x;
        //gate_2.y += gate_bias_2.y;
        gate_2 = GeluActivation<T2>::apply(gate_2);
        dst[cur_row * inner_col + cur_col]     = hidden_state_2.x * gate_2.x;
        dst[cur_row * inner_col + cur_col + 1] = hidden_state_2.y * gate_2.y;
    }
}



template<typename T, int32_t DIM_PER_THREAD>
__global__ void fusedAddBiasGegluV2(
    T* dst, const T* hidden_states, const T* bias, const int seq_len, const int dim, const int batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2
    int batch_idx = blockIdx.y;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int col     = threadIdx.x * 4 * DIM_PER_THREAD;

    using T4 = typename TypeConverter4<T>::Type;  // float to float4, half to half4
    using T2 = typename TypeConverter<T>::Type;   // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col = col + i * 4;
        if (cur_col > inner_col) {
            break;
        }
        T4 hidden_state_4      = *reinterpret_cast<const T4*>(hidden_states + cur_row * outer_col + cur_col);
        T4 hidden_state_bias_4 = *reinterpret_cast<const T4*>(bias + cur_col);
        T4 gate_4      = *reinterpret_cast<const T4*>(hidden_states + cur_row * outer_col + cur_col + inner_col);
        T4 gate_bias_4 = *reinterpret_cast<const T4*>(bias + cur_col + inner_col);

        T2 gate_2_1;
        T2 gate_2_2;

        gate_2_1.x = gate_4.x + gate_bias_4.x;
        gate_2_1.y = gate_4.y + gate_bias_4.y;
        gate_2_2.x = gate_4.z + gate_bias_4.z;
        gate_2_2.y = gate_4.w + gate_bias_4.w;

        gate_2_1 = GeluActivation<T2>::apply(gate_2_1);
        gate_2_2 = GeluActivation<T2>::apply(gate_2_2);

        T4 ele4;
        ele4.x = (hidden_state_4.x + hidden_state_bias_4.x) * gate_2_1.x;
        ele4.y = (hidden_state_4.y + hidden_state_bias_4.y) * gate_2_1.y;
        ele4.z = (hidden_state_4.z + hidden_state_bias_4.z) * gate_2_2.x;
        ele4.w = (hidden_state_4.w + hidden_state_bias_4.w) * gate_2_2.y;

        *reinterpret_cast<T4*>(&dst[cur_row * inner_col + cur_col]) = ele4;
    }
}

template<typename T>
void invokeFusedAddBiasGeglu(T*           dst,
                             const T*     hidden_states,
                             const T*     bias,
                             const int    seq_len,
                             const int    dim,
                             const int    batch_size,
                             cudaStream_t stream)
{
    // dim 的长度仅为 1280, 2560, 5120 的一种
    dim3 grid(seq_len, batch_size);
    switch (dim) {
        case 1280:
            if(bias!=nullptr){
                fusedAddBiasGeglu<T, 1><<<grid, 640, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            }
            else{
                fusedGeglu<T, 1><<<grid, 640, 0, stream>>>(dst, hidden_states, seq_len, dim, batch_size);
            }
            break;
        case 2560:
            if(bias!=nullptr){
                fusedAddBiasGeglu<T, 2><<<grid, 640, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            }
            else{
                fusedGeglu<T, 2><<<grid, 640, 0, stream>>>(dst, hidden_states, seq_len, dim, batch_size);
            }
            break;
        case 5120:
            if(bias!=nullptr){
                fusedAddBiasGeglu<T, 4><<<grid, 640, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            }
            else{
                fusedGeglu<T, 4><<<grid, 640, 0, stream>>>(dst, hidden_states, seq_len, dim, batch_size);
            }
            break;
        default:
            throw "unsupported dim size";
            break;
    }
}

template<typename T>
void invokeFusedAddBiasGegluV2(T*           dst,
                               const T*     hidden_states,
                               const T*     bias,
                               const int    seq_len,
                               const int    dim,
                               const int    batch_size,
                               cudaStream_t stream)
{
    // dim 的长度仅为 1280, 2560, 5120 的一种
    dim3 grid(seq_len, batch_size);
    switch (dim) {
        case 1280:
            fusedAddBiasGegluV2<T, 1><<<grid, 320, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            break;
        case 2560:
            fusedAddBiasGegluV2<T, 1><<<grid, 640, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            break;
        case 5120:
            fusedAddBiasGegluV2<T, 2><<<grid, 640, 0, stream>>>(dst, hidden_states, bias, seq_len, dim, batch_size);
            break;
        default:
            throw "unsupported dim size";
            break;
    }
}

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU(T)                                                                     \
    template void invokeFusedAddBiasGeglu(T*           dst,                                                            \
                                          const T*     hidden_states,                                                  \
                                          const T*     bias,                                                           \
                                          const int    seq_len,                                                        \
                                          const int    dim,                                                            \
                                          const int    batch_size,                                                     \
                                          cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU(half);
#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_V2(T)                                                                  \
    template void invokeFusedAddBiasGegluV2(T*           dst,                                                          \
                                            const T*     hidden_states,                                                \
                                            const T*     bias,                                                         \
                                            const int    seq_len,                                                      \
                                            const int    dim,                                                          \
                                            const int    batch_size,                                                   \
                                            cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_V2(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_V2(half);
#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_V2

}  // namespace lyradiff