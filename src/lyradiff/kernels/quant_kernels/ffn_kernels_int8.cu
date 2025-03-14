#include "ffn_kernels_int8.h"
#include "int8_utils.cuh"
#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

using namespace std;

namespace lyradiff {

template<typename T, int32_t DIM_PER_THREAD>
__global__ void fusedAddBiasGegluInt8OV2(int8_t*      dst,
                                         const T*     hidden_states,
                                         const T*     bias,
                                         const float* pre_quant_scale,
                                         const float* input_quant_scale,
                                         const int    seq_len,
                                         const int    dim,
                                         const int    batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2

    int col_batch_idx = blockIdx.y;
    int block_size    = blockDim.x;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD + col_batch_idx * block_size * 2 * DIM_PER_THREAD;

    float cur_input_quant_scale = __ldg(input_quant_scale);

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2
#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col             = col + i * 2;
        T2  hidden_state_2      = __ldg((const T2*)(hidden_states + cur_row * outer_col + cur_col));
        T2  hidden_state_bias_2 = __ldg((const T2*)(bias + cur_col));
        T2  gate_2              = __ldg((const T2*)(hidden_states + cur_row * outer_col + cur_col + inner_col));
        T2  gate_bias_2         = __ldg((const T2*)(bias + cur_col + inner_col));

        float2 pre_quant_scale_2 = __ldg((const float2*)(pre_quant_scale + cur_col));

        hidden_state_2.x += hidden_state_bias_2.x;
        hidden_state_2.y += hidden_state_bias_2.y;
        gate_2.x += gate_bias_2.x;
        gate_2.y += gate_bias_2.y;
        gate_2 = GeluActivation<T2>::apply(gate_2);

        dst[cur_row * inner_col + cur_col]     = float_to_int8_rn(cuda_cast<float>(hidden_state_2.x * gate_2.x)
                                                              * pre_quant_scale_2.x / cur_input_quant_scale);
        dst[cur_row * inner_col + cur_col + 1] = float_to_int8_rn(cuda_cast<float>(hidden_state_2.y * gate_2.y)
                                                                  * pre_quant_scale_2.y / cur_input_quant_scale);
    }
}

template<typename T, int32_t DIM_PER_THREAD>
__global__ void fusedGegluInt8OV2(int8_t*      dst,
                                  const T*     hidden_states,
                                  const float* pre_quant_scale,
                                  const float* input_quant_scale,
                                  const int    seq_len,
                                  const int    dim,
                                  const int    batch_size)
{
    // grid x y 维度分别代表 sql_len * batch, column batch = dim / 2 / 128
    // block x 为 128
    int col_batch_idx = blockIdx.y;
    int block_size    = blockDim.x;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD + col_batch_idx * block_size * 2 * DIM_PER_THREAD;

    float cur_input_quant_scale = __ldg(input_quant_scale);
#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col       = col + i * 2;
        using T2          = typename TypeConverter<T>::Type;  // float to float2, half to half2
        T2 hidden_state_2 = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col);
        T2 gate_2         = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col + inner_col);

        float2 pre_quant_scale_2 = __ldg((const float2*)(pre_quant_scale + cur_col));

        gate_2 = GeluActivation<T2>::apply(gate_2);

        dst[cur_row * inner_col + cur_col]     = float_to_int8_rn(cuda_cast<float>(hidden_state_2.x * gate_2.x)
                                                              * pre_quant_scale_2.x / cur_input_quant_scale);
        dst[cur_row * inner_col + cur_col + 1] = float_to_int8_rn(cuda_cast<float>(hidden_state_2.y * gate_2.y)
                                                                  * pre_quant_scale_2.y / cur_input_quant_scale);
    }
}

template<typename T, int32_t DIM_PER_THREAD>
__global__ void fusedAddBiasGegluInt8O(int8_t*      dst,
                                       const T*     hidden_states,
                                       const T*     bias,
                                       const float* pre_quant_scale,
                                       const float* input_quant_scale,
                                       const int    seq_len,
                                       const int    dim,
                                       const int    batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2
    int batch_idx = blockIdx.y;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD;

    float cur_input_quant_scale = __ldg(input_quant_scale);

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col             = col + i * 2;
        T2  hidden_state_2      = __ldg((const T2*)(hidden_states + cur_row * outer_col + cur_col));
        T2  hidden_state_bias_2 = __ldg((const T2*)(bias + cur_col));
        T2  gate_2              = __ldg((const T2*)(hidden_states + cur_row * outer_col + cur_col + inner_col));
        T2  gate_bias_2         = __ldg((const T2*)(bias + cur_col + inner_col));

        float2 pre_quant_scale_2 = __ldg((const float2*)(pre_quant_scale + cur_col));

        hidden_state_2.x += hidden_state_bias_2.x;
        hidden_state_2.y += hidden_state_bias_2.y;
        gate_2.x += gate_bias_2.x;
        gate_2.y += gate_bias_2.y;
        gate_2 = GeluActivation<T2>::apply(gate_2);

        dst[cur_row * inner_col + cur_col]     = float_to_int8_rn(cuda_cast<float>(hidden_state_2.x * gate_2.x)
                                                              * pre_quant_scale_2.x / cur_input_quant_scale);
        dst[cur_row * inner_col + cur_col + 1] = float_to_int8_rn(cuda_cast<float>(hidden_state_2.y * gate_2.y)
                                                                  * pre_quant_scale_2.y / cur_input_quant_scale);
    }
}

template<typename T, int32_t DIM_PER_THREAD>
__global__ void fusedGegluInt8O(int8_t*      dst,
                                const T*     hidden_states,
                                const float* pre_quant_scale,
                                const float* input_quant_scale,
                                const int    seq_len,
                                const int    dim,
                                const int    batch_size)
{
    // grid x y 维度分别代表 sql_len, batch id
    // block x 为 dim / 2
    int batch_idx = blockIdx.y;

    int outer_col = dim * 2;
    int inner_col = dim;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int col     = threadIdx.x * 2 * DIM_PER_THREAD;

    float cur_input_quant_scale = __ldg(input_quant_scale);

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int cur_col        = col + i * 2;
        T2  hidden_state_2 = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col);
        T2  gate_2         = *reinterpret_cast<const T2*>(hidden_states + cur_row * outer_col + cur_col + inner_col);

        float2 pre_quant_scale_2 = __ldg((const float2*)(pre_quant_scale + cur_col));

        gate_2 = GeluActivation<T2>::apply(gate_2);

        dst[cur_row * inner_col + cur_col]     = float_to_int8_rn(cuda_cast<float>(hidden_state_2.x * gate_2.x)
                                                              * pre_quant_scale_2.x / cur_input_quant_scale);
        dst[cur_row * inner_col + cur_col + 1] = float_to_int8_rn(cuda_cast<float>(hidden_state_2.y * gate_2.y)
                                                                  * pre_quant_scale_2.y / cur_input_quant_scale);
    }
}

template<typename T>
void invokeFusedAddBiasGegluInt8O(int8_t*      dst,
                                  const T*     hidden_states,
                                  const T*     bias,
                                  const float* pre_quant_scale,
                                  const float* input_quant_scale,
                                  const int    seq_len,
                                  const int    dim,
                                  const int    batch_size,
                                  cudaStream_t stream)
{
    // dim 的长度仅为 1280, 2560, 5120 的一种
    dim3 grid(seq_len, batch_size);
    switch (dim) {
        case 1280:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8O<T, 1><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8O<T, 1><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        case 2560:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8O<T, 2><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8O<T, 2><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        case 5120:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8O<T, 4><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8O<T, 4><<<grid, 640, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        default:
            throw "unsupported dim size";
            break;
    }
}

template<typename T>
void invokeFusedAddBiasGegluInt8OV2(int8_t*      dst,
                                    const T*     hidden_states,
                                    const T*     bias,
                                    const float* pre_quant_scale,
                                    const float* input_quant_scale,
                                    const int    seq_len,
                                    const int    dim,
                                    const int    batch_size,
                                    cudaStream_t stream)
{
    // dim 的长度仅为 1280, 2560, 5120 的一种

    size_t block_size = 128;
    dim3 grid(seq_len * batch_size, 5);

    switch (dim) {
        case 1280:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8OV2<T, 1><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8OV2<T, 1><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        case 2560:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8OV2<T, 2><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8O<T, 2><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        case 5120:
            if (bias != nullptr) {
                fusedAddBiasGegluInt8OV2<T, 4><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, bias, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            else {
                fusedGegluInt8OV2<T, 4><<<grid, block_size, 0, stream>>>(
                    dst, hidden_states, pre_quant_scale, input_quant_scale, seq_len, dim, batch_size);
            }
            break;
        default:
            throw "unsupported dim size";
            break;
    }
}

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8O(T)                                                               \
    template void invokeFusedAddBiasGegluInt8O(int8_t*      dst,                                                       \
                                               const T*     hidden_states,                                             \
                                               const T*     bias,                                                      \
                                               const float* pre_quant_scale,                                           \
                                               const float* input_quant_scale,                                         \
                                               const int    seq_len,                                                   \
                                               const int    dim,                                                       \
                                               const int    batch_size,                                                \
                                               cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8O(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8O(half);
#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8O

#define INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8OV2(T)                                                             \
    template void invokeFusedAddBiasGegluInt8OV2(int8_t*      dst,                                                     \
                                                 const T*     hidden_states,                                           \
                                                 const T*     bias,                                                    \
                                                 const float* pre_quant_scale,                                         \
                                                 const float* input_quant_scale,                                       \
                                                 const int    seq_len,                                                 \
                                                 const int    dim,                                                     \
                                                 const int    batch_size,                                              \
                                                 cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8OV2(float);
INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8OV2(half);
#undef INSTANTIATE_INVOKE_FUSED_ADD_BIAS_GEGLU_INT8OV2

}  // namespace lyradiff