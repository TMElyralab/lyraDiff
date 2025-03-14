#include "flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/reduce.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
__global__ void fuseCatAndGelu(
    T* out, const T* attn_output, const T* mlp_output, int batch_size, int seq_len, int attn_dim, int mlp_dim)
{
    using T2                   = typename TypeConverter<T>::Type;
    size_t cur_dst_col_offset  = blockIdx.x * (attn_dim + mlp_dim);
    size_t cur_attn_col_offset = blockIdx.x * attn_dim;
    size_t cur_mlp_col_offset  = blockIdx.x * mlp_dim;

    // 处理 attn 输出
    for (uint idx = threadIdx.x; idx * 2 < attn_dim; idx += blockDim.x) {
        size_t cur_dst_offset  = cur_dst_col_offset + idx * 2;
        size_t cur_attn_offset = cur_attn_col_offset + idx * 2;

        T2 cur_data = *reinterpret_cast<const T2*>(attn_output + cur_attn_offset);

        *reinterpret_cast<T2*>(&out[cur_dst_offset]) = cur_data;
    }

    // 处理 dst 输出
    for (uint idx = threadIdx.x; idx * 2 < mlp_dim; idx += blockDim.x) {
        size_t cur_dst_offset = cur_dst_col_offset + attn_dim + idx * 2;
        size_t cur_mlp_offset = cur_mlp_col_offset + idx * 2;

        T2 cur_data = *reinterpret_cast<const T2*>(mlp_output + cur_mlp_offset);

        // *reinterpret_cast<T2*>(&out[cur_dst_offset]) = GeluActivation<T2>::apply(cur_data);
        *reinterpret_cast<T2*>(&out[cur_dst_offset]) =
            cuda_cast<T2>(GeluActivation<float2>::apply(cuda_cast<float2>(cur_data)));
    }
}

template<typename T>
__global__ void
fuseGateAndResidual(T* out, const T* input, const T* gate, const T* residual, int batch_size, int seq_len, int dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_col_offset      = batch_idx * seq_len * dim + seq_idx * dim;
    size_t cur_gate_col_offset = batch_idx * dim;

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_offset      = cur_col_offset + idx * 2;
        size_t cur_gate_offset = cur_gate_col_offset + idx * 2;

        T2 cur_data          = *reinterpret_cast<const T2*>(input + cur_offset);
        T2 cur_gate_data     = *reinterpret_cast<const T2*>(gate + cur_gate_offset);
        T2 cur_residual_data = *reinterpret_cast<const T2*>(residual + cur_offset);

        cur_data = mul(cur_data, cur_gate_data);
        cur_data = add(cur_data, cur_residual_data);

        *reinterpret_cast<T2*>(&out[cur_offset]) = cur_data;
    }
}

template<typename T>
__global__ void
fusedBiasAndResidual(T* out, const T* input, const T* bias, const T* residual, int batch_size, int seq_len, int dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_col_offset = batch_idx * seq_len * dim + seq_idx * dim;
    // size_t cur_gate_col_offset = batch_idx * dim;

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_offset      = cur_col_offset + idx * 2;
        size_t cur_bias_offset = idx * 2;

        T2 cur_data          = *reinterpret_cast<const T2*>(input + cur_offset);
        T2 cur_bias_data     = *reinterpret_cast<const T2*>(bias + cur_bias_offset);
        T2 cur_residual_data = *reinterpret_cast<const T2*>(residual + cur_offset);

        cur_data = add(cur_data, cur_bias_data);
        cur_data = add(cur_data, cur_residual_data);

        *reinterpret_cast<T2*>(&out[cur_offset]) = cur_data;
    }
}

template<typename T>
__global__ void addBias(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_col_offset = batch_idx * seq_len * dim + seq_idx * dim;

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_offset      = cur_col_offset + idx * 2;
        size_t cur_bias_offset = idx * 2;

        T2 cur_data      = *reinterpret_cast<const T2*>(input + cur_offset);
        T2 cur_bias_data = *reinterpret_cast<const T2*>(bias + cur_bias_offset);

        cur_data                                 = add(cur_data, cur_bias_data);
        *reinterpret_cast<T2*>(&out[cur_offset]) = cur_data;
    }
}

template<typename T>
__global__ void fusedBiasAndGelu(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_col_offset = batch_idx * seq_len * dim + seq_idx * dim;

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_offset      = cur_col_offset + idx * 2;
        size_t cur_bias_offset = idx * 2;

        T2 cur_data      = *reinterpret_cast<const T2*>(input + cur_offset);
        T2 cur_bias_data = *reinterpret_cast<const T2*>(bias + cur_bias_offset);

        cur_data = add(cur_data, cur_bias_data);
        cur_data = cuda_cast<T2>(GeluActivation<float2>::apply(cuda_cast<float2>(cur_data)));
        *reinterpret_cast<T2*>(&out[cur_offset]) = cur_data;
    }
}

// 专用的flux single Transformer block中的算子，支持对mlp 输入的gelu，以及把attn output和mlp concat
template<typename T>
void invokeFusedCatAndGelu(T*           out,
                           const T*     attn_output,
                           const T*     mlp_output,
                           int          batch_size,
                           int          seq_len,
                           int          attn_dim,
                           int          mlp_dim,
                           cudaStream_t stream)
{
    dim3 grid(seq_len * batch_size);
    dim3 block(std::min(attn_dim / 2, 1024));
    fuseCatAndGelu<<<grid, block, 0, stream>>>(out, attn_output, mlp_output, batch_size, seq_len, attn_dim, mlp_dim);
}

template<typename T>
void invokeFusedGateAndResidual(
    T* out, const T* input, const T* gate, const T* residual, int batch_size, int seq_len, int dim, cudaStream_t stream)
{
    // 这里头维度乘2是为了只处理 qk 的数据
    dim3 grid(seq_len, batch_size);
    dim3 block(std::min(dim / 2, 1024));
    fuseGateAndResidual<<<grid, block, 0, stream>>>(out, input, gate, residual, batch_size, seq_len, dim);
}

template<typename T>
void invokeFusedBiasAndResidual(
    T* out, const T* input, const T* bias, const T* residual, int batch_size, int seq_len, int dim, cudaStream_t stream)
{
    // 这里头维度乘2是为了只处理 qk 的数据
    dim3 grid(seq_len, batch_size);
    dim3 block(std::min(dim / 2, 1024));
    fusedBiasAndResidual<<<grid, block, 0, stream>>>(out, input, bias, residual, batch_size, seq_len, dim);
}

template<typename T>
void invokeAddBias(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream)
{
    // 这里头维度乘2是为了只处理 qk 的数据
    dim3 grid(seq_len, batch_size);
    dim3 block(std::min(dim / 2, 1024));
    addBias<<<grid, block, 0, stream>>>(out, input, bias, batch_size, seq_len, dim);
}

template<typename T>
void invokeFusedBiasAndGelu(T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream)
{
    // 这里头维度乘2是为了只处理 qk 的数据
    dim3 grid(seq_len, batch_size);
    dim3 block(std::min(dim / 2, 1024));
    fusedBiasAndGelu<<<grid, block, 0, stream>>>(out, input, bias, batch_size, seq_len, dim);
}

#define INSTANTIATE_INVOKE_FUSED_CAT_AND_GELU(T)                                                                       \
    template void invokeFusedCatAndGelu(T*           out,                                                              \
                                        const T*     attn_output,                                                      \
                                        const T*     mlp_output,                                                       \
                                        int          batch_size,                                                       \
                                        int          seq_len,                                                          \
                                        int          attn_dim,                                                         \
                                        int          mlp_dim,                                                          \
                                        cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_CAT_AND_GELU(float);
INSTANTIATE_INVOKE_FUSED_CAT_AND_GELU(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_CAT_AND_GELU(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_CAT_AND_GELU

#define INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL(T)                                                                  \
    template void invokeFusedGateAndResidual(T*           out,                                                         \
                                             const T*     input,                                                       \
                                             const T*     gate,                                                        \
                                             const T*     residual,                                                    \
                                             int          batch_size,                                                  \
                                             int          seq_len,                                                     \
                                             int          dim,                                                         \
                                             cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL(float);
INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL

#define INSTANTIATE_INVOKE_FUSED_BIAS_AND_RESIDUAL(T)                                                                  \
    template void invokeFusedBiasAndResidual(T*           out,                                                         \
                                             const T*     input,                                                       \
                                             const T*     bias,                                                        \
                                             const T*     residual,                                                    \
                                             int          batch_size,                                                  \
                                             int          seq_len,                                                     \
                                             int          dim,                                                         \
                                             cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_BIAS_AND_RESIDUAL(float);
INSTANTIATE_INVOKE_FUSED_BIAS_AND_RESIDUAL(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_BIAS_AND_RESIDUAL(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL

#define INSTANTIATE_INVOKE_AND_BIAS(T)                                                                                 \
    template void invokeAddBias(                                                                                       \
        T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream)

INSTANTIATE_INVOKE_AND_BIAS(float);
INSTANTIATE_INVOKE_AND_BIAS(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_AND_BIAS(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_GATE_AND_RESIDUAL

#define INSTANTIATE_INVOKE_FUSED_BIAS_AND_GELU(T)                                                                      \
    template void invokeFusedBiasAndGelu(                                                                              \
        T* out, T* input, const T* bias, int batch_size, int seq_len, int dim, cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_BIAS_AND_GELU(float);
INSTANTIATE_INVOKE_FUSED_BIAS_AND_GELU(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_BIAS_AND_GELU(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_FUSED_BIAS_AND_GELU

}  // namespace lyradiff