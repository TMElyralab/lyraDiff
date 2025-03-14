#include "cuda_fp16.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include <iostream>

namespace lyradiff {

template<typename T, int32_t NUM_GROUP>
__global__ void attnLayerInputPermute(T*       dst,
                                      const T* src,
                                      size_t   batch_size,
                                      size_t   seq_len,
                                      size_t   dim,
                                      size_t   head_num,
                                      size_t   dim_per_head,
                                      size_t   offset        = 0,
                                      size_t   total_seq_len = 0)
{
    int batch_idx = blockIdx.y;

    int cur_row      = batch_idx * seq_len + blockIdx.x;
    int cur_col      = threadIdx.x * 2;
    int cur_head     = cur_col / dim_per_head;
    int cur_head_idx = cur_col % dim_per_head;

    int outer_col = dim * NUM_GROUP;
    int dst_row   = batch_idx * total_seq_len + blockIdx.x + offset;
    int dst_col   = cur_head * dim_per_head * NUM_GROUP + cur_head_idx;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // for self attention, this helps permute qkv res from gemm to input for attention layer
    // for corss attention, this helps permute kv res from gemm to input for attention layer
#pragma unroll
    for (int i = 0; i < NUM_GROUP; i++) {
        T2 data_2 = *reinterpret_cast<const T2*>(src + cur_row * outer_col + i * dim + cur_col);
        dst[dst_row * outer_col + dst_col + i * dim_per_head]     = data_2.x;
        dst[dst_row * outer_col + dst_col + i * dim_per_head + 1] = data_2.y;
    }
}

template<typename T, int32_t NUM_GROUP>
__global__ void attnLayerInputPermuteAndScaleK(T*       dst,
                                               const T* src,
                                               size_t   batch_size,
                                               size_t   seq_len,
                                               size_t   dim,
                                               size_t   head_num,
                                               size_t   dim_per_head,
                                               size_t   offset        = 0,
                                               size_t   total_seq_len = 0,
                                               T        scale_k       = 1)
{
    int batch_idx = blockIdx.y;

    int cur_row      = batch_idx * seq_len + blockIdx.x;
    int cur_col      = threadIdx.x * 2;
    int cur_head     = cur_col / dim_per_head;
    int cur_head_idx = cur_col % dim_per_head;

    int outer_col = dim * NUM_GROUP;
    int dst_row   = batch_idx * total_seq_len + blockIdx.x + offset;
    int dst_col   = cur_head * dim_per_head * NUM_GROUP + cur_head_idx;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // for self attention, this helps permute qkv res from gemm to input for attention layer
    // for corss attention, this helps permute kv res from gemm to input for attention layer
    T2 data_2k                             = *reinterpret_cast<const T2*>(src + cur_row * outer_col + cur_col);
    dst[dst_row * outer_col + dst_col]     = data_2k.x * scale_k;
    dst[dst_row * outer_col + dst_col + 1] = data_2k.y * scale_k;

    T2 data_2v = *reinterpret_cast<const T2*>(src + cur_row * outer_col + dim + cur_col);
    dst[dst_row * outer_col + dst_col + dim_per_head]     = data_2v.x;
    dst[dst_row * outer_col + dst_col + dim_per_head + 1] = data_2v.y;
}

template<typename T, int32_t NUM_GROUP>
__global__ void attnLayerInputPermuteAndScaleKPtr(T*       dst,
                                                  const T* src,
                                                  size_t   batch_size,
                                                  size_t   seq_len,
                                                  size_t   dim,
                                                  size_t   head_num,
                                                  size_t   dim_per_head,
                                                  size_t   offset,
                                                  size_t   total_seq_len,
                                                  size_t   scale_len,
                                                  float*   scale_k)
{
    int batch_idx = blockIdx.y;

    int cur_row      = batch_idx * seq_len + blockIdx.x;
    int cur_col      = threadIdx.x * 2;
    int cur_head     = cur_col / dim_per_head;
    int cur_head_idx = cur_col % dim_per_head;

    int outer_col = dim * NUM_GROUP;
    int dst_row   = batch_idx * total_seq_len + blockIdx.x + offset;
    int dst_col   = cur_head * dim_per_head * NUM_GROUP + cur_head_idx;

    float cur_scale = scale_k[0];

    T scale = cuda_cast<T>(cur_scale);

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // for self attention, this helps permute qkv res from gemm to input for attention layer
    // for corss attention, this helps permute kv res from gemm to input for attention layer
    T2 data_2k                             = *reinterpret_cast<const T2*>(src + cur_row * outer_col + cur_col);
    dst[dst_row * outer_col + dst_col]     = data_2k.x * scale;
    dst[dst_row * outer_col + dst_col + 1] = data_2k.y * scale;

    T2 data_2v = *reinterpret_cast<const T2*>(src + cur_row * outer_col + dim + cur_col);
    dst[dst_row * outer_col + dst_col + dim_per_head]     = data_2v.x;
    dst[dst_row * outer_col + dst_col + dim_per_head + 1] = data_2v.y;
}

template<typename T, int32_t NUM_GROUP>
__global__ void attnLayerInputPermuteAndScaleKPtr2(T*       dst,
                                                   const T* src,
                                                   size_t   batch_size,
                                                   size_t   seq_len,
                                                   size_t   dim,
                                                   size_t   head_num,
                                                   size_t   dim_per_head,
                                                   size_t   offset,
                                                   size_t   total_seq_len,
                                                   size_t   scale_len,
                                                   float*   scale_k)
{
    int batch_idx = blockIdx.y;

    int cur_row      = batch_idx * seq_len + blockIdx.x;
    int cur_col      = threadIdx.x * 2;
    int cur_head     = cur_col / dim_per_head;
    int cur_head_idx = cur_col % dim_per_head;

    int outer_col = dim * NUM_GROUP;
    int dst_row   = batch_idx * total_seq_len + blockIdx.x + offset;
    int dst_col   = cur_head * dim_per_head * NUM_GROUP + cur_head_idx;

    float cur_scale = scale_k[blockIdx.x];

    T scale = cuda_cast<T>(cur_scale);

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // for self attention, this helps permute qkv res from gemm to input for attention layer
    // for corss attention, this helps permute kv res from gemm to input for attention layer
    T2 data_2k                             = *reinterpret_cast<const T2*>(src + cur_row * outer_col + cur_col);
    dst[dst_row * outer_col + dst_col]     = data_2k.x * scale;
    dst[dst_row * outer_col + dst_col + 1] = data_2k.y * scale;

    T2 data_2v = *reinterpret_cast<const T2*>(src + cur_row * outer_col + dim + cur_col);
    dst[dst_row * outer_col + dst_col + dim_per_head]     = data_2v.x;
    dst[dst_row * outer_col + dst_col + dim_per_head + 1] = data_2v.y;
}

template<typename T, int32_t NUM_GROUP>
__global__ void attn2LayerInputPermute(T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group)
{
    int row = blockIdx.x;
    int col = threadIdx.x * 4;

    int col_size = dim_per_group * NUM_GROUP;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

#pragma unroll
    for (int i = 0; i < NUM_GROUP; i++) {
        int cur_col = col + i * dim_per_group;
        int dst_row = i * batch_size * seq_len + row;
        int dst_col = col;

        T4 ele4 = *reinterpret_cast<const T4*>(src + row * col_size + cur_col);
        *reinterpret_cast<T4*>(&dst[dst_row * dim_per_group + dst_col]) = ele4;
    }
}

template<typename T>
__global__ void fusedBiasResidualAdd(
    T* dst, const T* src, const T* residual, const T* bias, size_t batch_size, size_t seq_len, size_t dim)
{
    // int batch_idx = blockIdx.y;

    // int cur_row = batch_idx * seq_len + blockIdx.x;
    // int cur_col = threadIdx.x * 2;

    // using T2 = typename TypeConverter<T>::Type; // float to float2, half to half2

    // T2 residual_2 = *reinterpret_cast<const T2 *>(residual + cur_row * dim + cur_col);
    // T2 src_2 = *reinterpret_cast<const T2 *>(src + cur_row * dim + cur_col);
    // T2 bias_2 = *reinterpret_cast<const T2 *>(bias + cur_col);

    // dst[cur_row * dim + cur_col] = src_2.x + residual_2.x + bias_2.x;
    // dst[cur_row * dim + cur_col + 1] = src_2.y + residual_2.y + bias_2.y;

    int batch_idx = blockIdx.y;

    int cur_row = batch_idx * seq_len + blockIdx.x;
    int cur_col = threadIdx.x * 4;

    using T4 = typename TypeConverter4<T>::Type;  // float to float4, half to half4

    T4 residual_4 = *reinterpret_cast<const T4*>(residual + cur_row * dim + cur_col);
    T4 src_4      = *reinterpret_cast<const T4*>(src + cur_row * dim + cur_col);
    T4 bias_4     = *reinterpret_cast<const T4*>(bias + cur_col);

    T4 ele4;
    ele4.x = src_4.x + residual_4.x + bias_4.x;
    ele4.y = src_4.y + residual_4.y + bias_4.y;
    ele4.z = src_4.z + residual_4.z + bias_4.z;
    ele4.w = src_4.w + residual_4.w + bias_4.w;

    *reinterpret_cast<T4*>(&dst[cur_row * dim + cur_col]) = ele4;
}

// attention layer 前处理，(batch_size, seq_len, 3, head_num, dim_per_head) -> (batch_size, seq_len, head_num, 3,
// dim_per_head)
template<typename T>
void invokeSelfAttnKernelInputPermute(T*           dst,
                                      const T*     src,
                                      size_t       batch_size,
                                      size_t       seq_len,
                                      size_t       dim,
                                      size_t       head_num,
                                      size_t       dim_per_head,
                                      cudaStream_t stream)
{
    // dim 仅为 320, 640, 1280
    dim3 block(dim / 2);
    dim3 grid(seq_len, batch_size);
    attnLayerInputPermute<T, 3>
        <<<grid, block, 0, stream>>>(dst, src, batch_size, seq_len, dim, head_num, dim_per_head, 0, seq_len);
}

template<typename T>
void invokeCrossAttnKernelInputPermute(T*           dst,
                                       const T*     src,
                                       size_t       batch_size,
                                       size_t       seq_len,
                                       size_t       dim,
                                       size_t       head_num,
                                       size_t       dim_per_head,
                                       cudaStream_t stream)
{
    // dim 仅为 320, 640, 1280
    dim3 block(dim / 2);
    dim3 grid(seq_len, batch_size);
    attnLayerInputPermute<T, 2>
        <<<grid, block, 0, stream>>>(dst, src, batch_size, seq_len, dim, head_num, dim_per_head, 0, seq_len);
}

template<typename T>
void invokeCrossAttnKernelInputPermuteWithOffset(T*           dst,
                                                 const T*     src,
                                                 size_t       batch_size,
                                                 size_t       seq_len,
                                                 size_t       dim,
                                                 size_t       head_num,
                                                 size_t       dim_per_head,
                                                 size_t       offset,
                                                 size_t       total_seq_len,
                                                 float        scale_k,
                                                 cudaStream_t stream)
{
    // dim 仅为 320, 640, 1280
    dim3 block(dim / 2);
    dim3 grid(seq_len, batch_size);
    if (float(scale_k) == 1.0) {
        attnLayerInputPermute<T, 2><<<grid, block, 0, stream>>>(
            dst, src, batch_size, seq_len, dim, head_num, dim_per_head, offset, total_seq_len);
    }
    else {
        attnLayerInputPermuteAndScaleK<T, 2><<<grid, block, 0, stream>>>(
            dst, src, batch_size, seq_len, dim, head_num, dim_per_head, offset, total_seq_len, __float2half(scale_k));
    }
}

template<typename T>
void invokeCrossAttnKernelInputPermuteAndScalePtrWithOffset(T*           dst,
                                                            const T*     src,
                                                            size_t       batch_size,
                                                            size_t       seq_len,
                                                            size_t       dim,
                                                            size_t       head_num,
                                                            size_t       dim_per_head,
                                                            size_t       offset,
                                                            size_t       total_seq_len,
                                                            size_t       scale_len,
                                                            float*       scale_k,
                                                            cudaStream_t stream)
{
    dim3 block(dim / 2);
    dim3 grid(seq_len, batch_size);

    if (scale_len == 1) {
        attnLayerInputPermuteAndScaleKPtr<T, 2><<<grid, block, 0, stream>>>(
            dst, src, batch_size, seq_len, dim, head_num, dim_per_head, offset, total_seq_len, scale_len, scale_k);
    }
    else {
        attnLayerInputPermuteAndScaleKPtr2<T, 2><<<grid, block, 0, stream>>>(
            dst, src, batch_size, seq_len, dim, head_num, dim_per_head, offset, total_seq_len, scale_len, scale_k);
    }
}

template<typename T, int32_t NUM_GROUP>
__global__ void attn2LayerInputPermuteAndScaleK(T*       dst,
                                                const T* src,
                                                size_t   batch_size,
                                                size_t   seq_len,
                                                size_t   dim_per_group,
                                                size_t   offset,
                                                size_t   total_seq_len,
                                                T        scale_k)
{
    int row = blockIdx.y * seq_len + blockIdx.x;
    int col = threadIdx.x << 2;

    int col_size = dim_per_group * NUM_GROUP;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

    int dstrow_idx = (blockIdx.y * total_seq_len + blockIdx.x + offset);  // x * total_seq_len + y;
    // for k
    if (blockIdx.z == 0) {
        int cur_col = col + blockIdx.z * dim_per_group;
        int dst_row = dstrow_idx + blockIdx.z * total_seq_len;
        int dst_col = col;

        T4 ele4 = *reinterpret_cast<const T4*>(src + row * col_size + cur_col);
        ele4.x *= scale_k;
        ele4.y *= scale_k;
        ele4.z *= scale_k;
        ele4.w *= scale_k;
        *reinterpret_cast<T4*>(&dst[dst_row * dim_per_group + dst_col]) = ele4;
    }
    else
    // for v
    {
        int cur_col = col + dim_per_group;
        int dst_row = batch_size * total_seq_len + dstrow_idx;
        int dst_col = col;

        *reinterpret_cast<T4*>(&dst[dst_row * dim_per_group + dst_col]) =
            *reinterpret_cast<const T4*>(src + row * col_size + cur_col);
    }
}

template<typename T, int32_t NUM_GROUP>
__global__ void attn2LayerInputPermuteAndScaleKPtr(T*       dst,
                                                   const T* src,
                                                   size_t   batch_size,
                                                   size_t   seq_len,
                                                   size_t   dim_per_group,
                                                   size_t   offset,
                                                   size_t   total_seq_len,
                                                   size_t   scale_len,
                                                   float*   scale_k_ptr)
{
    // batchsize, seq_len, 2 , dims -> batchsize, seq_len,
    // 原来： 2*dim  新： dim
    int row = blockIdx.y * seq_len + blockIdx.x;
    int col = threadIdx.x << 2;

    int col_size = dim_per_group * NUM_GROUP;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

    int dstrow_idx = (blockIdx.y * total_seq_len + blockIdx.x + offset);  // x * total_seq_len + y;

    float cur_scale;
    cur_scale = scale_k_ptr[blockIdx.x];
    T scale_k = cuda_cast<T>(cur_scale);

    // for k
    if (blockIdx.z == 0) {
        int cur_col = col + blockIdx.z * dim_per_group;
        int dst_row = dstrow_idx + blockIdx.z * total_seq_len;
        int dst_col = col;

        T4 ele4 = *reinterpret_cast<const T4*>(src + row * col_size + cur_col);
        ele4.x *= scale_k;
        ele4.y *= scale_k;
        ele4.z *= scale_k;
        ele4.w *= scale_k;
        *reinterpret_cast<T4*>(&dst[dst_row * dim_per_group + dst_col]) = ele4;
    }
    else
    // for v
    {
        int cur_col = col + dim_per_group;
        int dst_row = batch_size * total_seq_len + dstrow_idx;
        int dst_col = col;

        *reinterpret_cast<T4*>(&dst[dst_row * dim_per_group + dst_col]) =
            *reinterpret_cast<const T4*>(src + row * col_size + cur_col);
    }
}

template<typename T>
__global__ void fusedIpMaskAndAddResidual(T*           dst,
                                          const T*     attention_res,
                                          const T*     ip_res,
                                          const T*     ip_region_mask,
                                          const float  scale,
                                          const size_t channel,
                                          const size_t seq_len,
                                          const size_t batch_size)
{
    int row = blockIdx.y * seq_len + blockIdx.x;
    int col = threadIdx.x << 2;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

    T cur_ip_mask = ip_region_mask[blockIdx.x];
    T cur_scale   = cuda_cast<T>(scale);

    T4 attention_res_4 = *reinterpret_cast<const T4*>(attention_res + row * channel + col);
    T4 ip_res_4        = *reinterpret_cast<const T4*>(ip_res + row * channel + col);

    T4 ele4;
    ele4.x = attention_res_4.x + ip_res_4.x * cur_ip_mask * cur_scale;
    ele4.y = attention_res_4.y + ip_res_4.y * cur_ip_mask * cur_scale;
    ele4.z = attention_res_4.z + ip_res_4.z * cur_ip_mask * cur_scale;
    ele4.w = attention_res_4.w + ip_res_4.w * cur_ip_mask * cur_scale;

    *reinterpret_cast<T4*>(&dst[row * channel + col]) = ele4;
}

template<typename T>
void invokeCrossAttn2KernelInputPermuteWithOffset(T*           dst,
                                                  const T*     src,
                                                  size_t       batch_size,
                                                  size_t       seq_len,
                                                  size_t       dim_per_group,
                                                  size_t       offset,
                                                  size_t       total_seq_len,
                                                  float        scale_k,
                                                  cudaStream_t stream)
{
    int block = dim_per_group / 4;

    dim3 grid(seq_len, batch_size, 2);
    attn2LayerInputPermuteAndScaleK<T, 2><<<grid, block, 0, stream>>>(
        dst, src, batch_size, seq_len, dim_per_group, offset, total_seq_len, __float2half(scale_k));
}

template<typename T>
void invokeCrossAttn2KernelInputPermuteAndScalePtrWithOffset(T*           dst,
                                                             const T*     src,
                                                             size_t       batch_size,
                                                             size_t       seq_len,
                                                             size_t       dim_per_group,
                                                             size_t       offset,
                                                             size_t       total_seq_len,
                                                             size_t       scale_len,
                                                             float*       scale_k,
                                                             cudaStream_t stream)
{
    int block = dim_per_group / 4;

    dim3 grid(seq_len, batch_size, 2);
    attn2LayerInputPermuteAndScaleKPtr<T, 2><<<grid, block, 0, stream>>>(
        dst, src, batch_size, seq_len, dim_per_group, offset, total_seq_len, scale_len, scale_k);
}

template<typename T>
void invokeFusedBiasResidualAdd(T*           dst,
                                const T*     src,
                                const T*     residual,
                                const T*     bias,
                                size_t       batch_size,
                                size_t       seq_len,
                                size_t       dim,
                                cudaStream_t stream)
{
    // dim 仅为 320, 640, 1280
    dim3 block(dim / 4);
    dim3 grid(seq_len, batch_size);
    fusedBiasResidualAdd<<<grid, block, 0, stream>>>(dst, src, residual, bias, batch_size, seq_len, dim);
}

template<typename T>
void invokeSelfAttn2KernelInputPermute(
    T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream)
{
    int block = dim_per_group / 4;
    int grid  = batch_size * seq_len;

    attn2LayerInputPermute<T, 3><<<grid, block, 0, stream>>>(dst, src, batch_size, seq_len, dim_per_group);
}

template<typename T>
void invokeCrossAttn2KernelInputPermute(
    T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream)
{
    int block = dim_per_group / 4;
    int grid  = batch_size * seq_len;

    attn2LayerInputPermute<T, 2><<<grid, block, 0, stream>>>(dst, src, batch_size, seq_len, dim_per_group);
}

template<typename T>
void invokeIpMaskAndAddResidual(T*           dst,
                                const T*     attention_res,
                                const T*     ip_res,
                                const T*     ip_region_mask,
                                const float  ip_scale,
                                const size_t channel,
                                const size_t seq_len,
                                const size_t batch_size,
                                cudaStream_t stream)
{
    // int block = dim_per_group / 4;
    // int grid  = batch_size * seq_len;

    dim3 block(channel / 4);
    dim3 grid(seq_len, batch_size);

    fusedIpMaskAndAddResidual<T><<<grid, block, 0, stream>>>(
        dst, attention_res, ip_res, ip_region_mask, ip_scale, channel, seq_len, batch_size);
}

#define INSTANTIATE_INVOKE_IP_MASK_AND_ADD_RESIDUAL(T)                                                                 \
    template void invokeIpMaskAndAddResidual(T*           dst,                                                         \
                                             const T*     attention_res,                                               \
                                             const T*     ip_res,                                                      \
                                             const T*     ip_region_mask,                                              \
                                             const float  ip_scale,                                                    \
                                             const size_t channel,                                                     \
                                             const size_t seq_len,                                                     \
                                             const size_t batch_size,                                                  \
                                             cudaStream_t stream)

INSTANTIATE_INVOKE_IP_MASK_AND_ADD_RESIDUAL(float);
INSTANTIATE_INVOKE_IP_MASK_AND_ADD_RESIDUAL(half);
#undef INSTANTIATE_INVOKE_IP_MASK_AND_ADD_RESIDUAL

#define INSTANTIATE_INVOKE_SELF_ATTN_KERNEL_INPUT_PERMUTE(T)                                                           \
    template void invokeSelfAttnKernelInputPermute(T*           dst,                                                   \
                                                   const T*     src,                                                   \
                                                   size_t       batch_size,                                            \
                                                   size_t       seq_len,                                               \
                                                   size_t       dim,                                                   \
                                                   size_t       head_num,                                              \
                                                   size_t       dim_per_head,                                          \
                                                   cudaStream_t stream)

INSTANTIATE_INVOKE_SELF_ATTN_KERNEL_INPUT_PERMUTE(float);
INSTANTIATE_INVOKE_SELF_ATTN_KERNEL_INPUT_PERMUTE(half);
#undef INSTANTIATE_INVOKE_SELF_ATTN_KERNEL_INPUT_PERMUTE

#define INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE(T)                                                          \
    template void invokeCrossAttnKernelInputPermute(T*           dst,                                                  \
                                                    const T*     src,                                                  \
                                                    size_t       batch_size,                                           \
                                                    size_t       seq_len,                                              \
                                                    size_t       dim,                                                  \
                                                    size_t       head_num,                                             \
                                                    size_t       dim_per_head,                                         \
                                                    cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE(float);
INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE(half);
#undef INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE

#define INSTANTIATE_INVOKE_FUSED_BIAS_FACTOR_RESIDUAL_ADD(T)                                                           \
    template void invokeFusedBiasResidualAdd(T*           dst,                                                         \
                                             const T*     src,                                                         \
                                             const T*     residual,                                                    \
                                             const T*     bias,                                                        \
                                             size_t       batch_size,                                                  \
                                             size_t       seq_len,                                                     \
                                             size_t       dim,                                                         \
                                             cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_BIAS_FACTOR_RESIDUAL_ADD(float);
INSTANTIATE_INVOKE_FUSED_BIAS_FACTOR_RESIDUAL_ADD(half);
#undef INSTANTIATE_INVOKE_FUSED_BIAS_FACTOR_RESIDUAL_ADD

#define INSTANTIATE_INVOKE_SELF_ATTN2_KERNEL_INPUT_PERMUTE(T)                                                          \
    template void invokeSelfAttn2KernelInputPermute(                                                                   \
        T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream)

INSTANTIATE_INVOKE_SELF_ATTN2_KERNEL_INPUT_PERMUTE(float);
INSTANTIATE_INVOKE_SELF_ATTN2_KERNEL_INPUT_PERMUTE(half);
#undef INSTANTIATE_INVOKE_SELF_ATTN2_KERNEL_INPUT_PERMUTE

#define INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE(T)                                                         \
    template void invokeCrossAttn2KernelInputPermute(                                                                  \
        T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE(float);
INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE(half);

#define INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_WITH_OFFSET(T)                                              \
    template void invokeCrossAttnKernelInputPermuteWithOffset(T*           dst,                                        \
                                                              const T*     src,                                        \
                                                              size_t       batch_size,                                 \
                                                              size_t       seq_len,                                    \
                                                              size_t       dim,                                        \
                                                              size_t       head_num,                                   \
                                                              size_t       dim_per_head,                               \
                                                              size_t       offset,                                     \
                                                              size_t       total_seq_len,                              \
                                                              float        scale_k,                                    \
                                                              cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_WITH_OFFSET(float);
INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_WITH_OFFSET(half);
#undef INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_WITH_OFFSET

#define INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(T)                                    \
    template void invokeCrossAttnKernelInputPermuteAndScalePtrWithOffset(T*           dst,                             \
                                                                         const T*     src,                             \
                                                                         size_t       batch_size,                      \
                                                                         size_t       seq_len,                         \
                                                                         size_t       dim,                             \
                                                                         size_t       head_num,                        \
                                                                         size_t       dim_per_head,                    \
                                                                         size_t       offset,                          \
                                                                         size_t       total_seq_len,                   \
                                                                         size_t       scale_len,                       \
                                                                         float*       scale_k,                         \
                                                                         cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(float);
INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(half);
#undef INSTANTIATE_INVOKE_CROSS_ATTN_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET

#define INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_WITH_OFFSET(T)                                             \
    template void invokeCrossAttn2KernelInputPermuteWithOffset(T*           dst,                                       \
                                                               const T*     src,                                       \
                                                               size_t       batch_size,                                \
                                                               size_t       seq_len,                                   \
                                                               size_t       dim_per_group,                             \
                                                               size_t       offset,                                    \
                                                               size_t       total_seq_len,                             \
                                                               float        scale_k,                                   \
                                                               cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_WITH_OFFSET(float);
INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_WITH_OFFSET(half);

#define INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(T)                                   \
    template void invokeCrossAttn2KernelInputPermuteAndScalePtrWithOffset(T*           dst,                            \
                                                                          const T*     src,                            \
                                                                          size_t       batch_size,                     \
                                                                          size_t       seq_len,                        \
                                                                          size_t       dim_per_group,                  \
                                                                          size_t       offset,                         \
                                                                          size_t       total_seq_len,                  \
                                                                          size_t       scale_len,                      \
                                                                          float*       scale_k,                        \
                                                                          cudaStream_t stream)

INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(float);
INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET(half);
#undef INSTANTIATE_INVOKE_CROSS_ATTN2_KERNEL_INPUT_PERMUTE_AND_SCALE_WITH_OFFSET

}  // namespace lyradiff
