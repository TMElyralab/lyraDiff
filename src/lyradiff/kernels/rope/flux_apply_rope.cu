#include "flux_apply_rope.h"
#include "src/lyradiff/reduce.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
__global__ void fuseRmsNormAndRope(T*       out,
                                   const T* input,
                                   const T* scale,
                                   const T* rope_emb,
                                   float    eps,
                                   int      batch_size,
                                   int      seq_len,
                                   int      head_num,
                                   int      head_dim)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int seq_idx   = blockIdx.x;
    int offset = batch_idx * seq_len * 3 * head_num * head_dim + seq_idx * 3 * head_num * head_dim + head_idx * head_dim
                 + threadIdx.x * 2;

    T2 cur_data = *reinterpret_cast<const T2*>(input + offset);
    // T2 cur_data1;
    if (head_idx >= head_num * 2) {
        *reinterpret_cast<T2*>(&out[offset]) = cur_data;
        return;
    }

    int rope_offset  = batch_idx * seq_len * head_dim * 2 + seq_idx * head_dim * 2 + threadIdx.x * 4;
    int scale_offset = threadIdx.x * 2;
    // 如果head_idx 到k，scale也需要到k才可以
    if (head_idx >= head_num) {
        scale_offset += head_dim;
    }

    T2 cur_scale  = *reinterpret_cast<const T2*>(scale + scale_offset);
    T2 cur_rope_0 = *reinterpret_cast<const T2*>(rope_emb + rope_offset);
    T2 cur_rope_1 = *reinterpret_cast<const T2*>(rope_emb + rope_offset + 2);

    float2 tmp2 = cuda_cast<float2>(cur_data);
    mean += tmp2.x * tmp2.x;
    mean += tmp2.y * tmp2.y;

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(mean / (float)head_dim + eps);
    }
    __syncthreads();

    tmp2.x = tmp2.x * s_inv_mean;
    tmp2.y = tmp2.y * s_inv_mean;

    T2 t2 = cuda_cast<T2>(tmp2);
    T2 ele2;

    t2.x = t2.x * cur_scale.x;
    t2.y = t2.y * cur_scale.y;

    ele2.x = t2.x * cur_rope_0.x + t2.y * cur_rope_0.y;
    ele2.y = t2.x * cur_rope_1.x + t2.y * cur_rope_1.y;

    *reinterpret_cast<T2*>(&out[offset]) = ele2;
}

template<typename T>
__global__ void fuseRmsNormCatAndRope(T*       out,
                                      const T* hidden,
                                      const T* encoder_hidden,
                                      const T* hidden_rms_scale,
                                      const T* encoder_rms_scale,
                                      const T* rope_emb,
                                      float    eps,
                                      int      batch_size,
                                      int      hidden_seq_len,
                                      int      encoder_seq_len,
                                      int      head_num,
                                      int      head_dim)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    using T2         = typename TypeConverter<T>::Type;
    size_t batch_idx = blockIdx.z;
    size_t head_idx  = blockIdx.y;
    size_t seq_idx   = blockIdx.x;

    size_t total_seq_len = hidden_seq_len + encoder_seq_len;

    size_t output_offset = batch_idx * total_seq_len * 3 * head_num * head_dim + seq_idx * 3 * head_num * head_dim
                           + head_idx * head_dim + threadIdx.x * 2;

    T2       cur_data;
    const T* scale = encoder_rms_scale;
    if (seq_idx < encoder_seq_len) {  // cat [encoder, hidden]，小于encoder_seq_len 的情况下，放置encoder数据
        size_t input_offset = batch_idx * encoder_seq_len * 3 * head_num * head_dim + seq_idx * 3 * head_num * head_dim
                              + head_idx * head_dim + threadIdx.x * 2;
        cur_data = *reinterpret_cast<const T2*>(encoder_hidden + input_offset);
    }
    else {  // cat [encoder, hidden]，大于等于 encoder_seq_len 的情况下，计算hidden数据
        size_t cur_seq_idx  = seq_idx - encoder_seq_len;
        size_t input_offset = batch_idx * hidden_seq_len * 3 * head_num * head_dim
                              + cur_seq_idx * 3 * head_num * head_dim + head_idx * head_dim + threadIdx.x * 2;
        cur_data = *reinterpret_cast<const T2*>(hidden + input_offset);
        scale    = hidden_rms_scale;
    }

    // 无论是hidden还是encoder，v的数据都不需要处理
    if (head_idx >= head_num * 2) {
        *reinterpret_cast<T2*>(&out[output_offset]) = cur_data;
        return;
    }

    size_t rope_offset  = batch_idx * total_seq_len * head_dim * 2 + seq_idx * head_dim * 2 + threadIdx.x * 4;
    size_t scale_offset = threadIdx.x * 2;
    // 如果head_idx 到k，scale也需要到k才可以
    if (head_idx >= head_num) {
        scale_offset += head_dim;
    }

    T2 cur_scale  = *reinterpret_cast<const T2*>(scale + scale_offset);
    T2 cur_rope_0 = *reinterpret_cast<const T2*>(rope_emb + rope_offset);
    T2 cur_rope_1 = *reinterpret_cast<const T2*>(rope_emb + rope_offset + 2);

    float2 tmp2 = cuda_cast<float2>(cur_data);
    mean += tmp2.x * tmp2.x;
    mean += tmp2.y * tmp2.y;

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(mean / (float)head_dim + eps);
    }
    __syncthreads();

    tmp2.x = tmp2.x * s_inv_mean;
    tmp2.y = tmp2.y * s_inv_mean;

    T2 t2 = cuda_cast<T2>(tmp2);
    T2 ele2;

    t2.x = t2.x * cur_scale.x;
    t2.y = t2.y * cur_scale.y;

    ele2.x = t2.x * cur_rope_0.x + t2.y * cur_rope_0.y;
    ele2.y = t2.x * cur_rope_1.x + t2.y * cur_rope_1.y;

    *reinterpret_cast<T2*>(&out[output_offset]) = ele2;
}

template<typename T>
__global__ void
fluxApplyRope(T* out, const T* input, const T* rope_emb, int batch_size, int seq_len, int head_num, int head_dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int seq_idx   = blockIdx.x;
    int offset    = batch_idx * seq_len * head_num * head_dim + seq_idx * head_num * head_dim + head_idx * head_dim
                 + threadIdx.x * 2;
    int rope_offset = seq_idx * head_dim * 2 + threadIdx.x * 4;

    T2 cur_data   = *reinterpret_cast<const T2*>(input + offset);
    T2 cur_rope_0 = *reinterpret_cast<const T2*>(rope_emb + rope_offset);
    T2 cur_rope_1 = *reinterpret_cast<const T2*>(rope_emb + rope_offset + 2);

    T2 ele2;
    ele2.x = cur_data.x * cur_rope_0.x + cur_data.y * cur_rope_0.y;
    ele2.y = cur_data.x * cur_rope_1.x + cur_data.y * cur_rope_1.y;

    *reinterpret_cast<T2*>(&out[offset]) = ele2;
}


template<typename T>
__global__ void fuseBiasRmsNormAndRope(T*       out,
                                       const T* input,
                                       const T* bias,
                                       const T* scale,
                                       const T* rope_emb,
                                       float    eps,
                                       int      batch_size,
                                       int      seq_len,
                                       int      head_num,
                                       int      head_dim)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.z;
    int head_idx  = blockIdx.y;
    int seq_idx   = blockIdx.x;
    int offset = batch_idx * seq_len * 3 * head_num * head_dim + seq_idx * 3 * head_num * head_dim + head_idx * head_dim
                 + threadIdx.x * 2;
    int gb_offset = head_idx * head_dim + threadIdx.x * 2;

    T2 cur_data = *reinterpret_cast<const T2*>(input + offset);
    T2 cur_bias = *reinterpret_cast<const T2*>(bias + gb_offset);
    cur_data    = add(cur_data, cur_bias);
    // T2 cur_data1;
    if (head_idx >= head_num * 2) {
        *reinterpret_cast<T2*>(&out[offset]) = cur_data;
        return;
    }

    int rope_offset  = batch_idx * seq_len * head_dim * 2 + seq_idx * head_dim * 2 + threadIdx.x * 4;
    int scale_offset = threadIdx.x * 2;
    // 如果head_idx 到k，scale也需要到k才可以
    if (head_idx >= head_num) {
        scale_offset += head_dim;
    }

    T2 cur_scale  = *reinterpret_cast<const T2*>(scale + scale_offset);
    T2 cur_rope_0 = *reinterpret_cast<const T2*>(rope_emb + rope_offset);
    T2 cur_rope_1 = *reinterpret_cast<const T2*>(rope_emb + rope_offset + 2);

    float2 tmp2 = cuda_cast<float2>(cur_data);
    mean += tmp2.x * tmp2.x;
    mean += tmp2.y * tmp2.y;

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(mean / (float)head_dim + eps);
    }
    __syncthreads();

    tmp2.x = tmp2.x * s_inv_mean;
    tmp2.y = tmp2.y * s_inv_mean;

    T2 t2 = cuda_cast<T2>(tmp2);
    T2 ele2;

    t2.x = t2.x * cur_scale.x;
    t2.y = t2.y * cur_scale.y;

    ele2.x = t2.x * cur_rope_0.x + t2.y * cur_rope_0.y;
    ele2.y = t2.x * cur_rope_1.x + t2.y * cur_rope_1.y;

    *reinterpret_cast<T2*>(&out[offset]) = ele2;
}


template<typename T>
void invokeFluxApplyRope(T*           out,
                         const T*     input,
                         const T*     rope_emb,
                         int          batch_size,
                         int          seq_len,
                         int          head_num,
                         int          head_dim,
                         cudaStream_t stream)
{
    dim3 grid(seq_len, head_num, batch_size);
    dim3 block(std::min(head_dim / 2, 1024));
    fluxApplyRope<<<grid, block, 0, stream>>>(out, input, rope_emb, batch_size, seq_len, head_num, head_dim);
}

// 这个kernel主要为了flux single transformers attention服务
// 主要针对single transformers attention 中 to_qkv 之后的 只针对 qk 做的 rmsnorm 和 rope 操作，这样方便直接塞给attention
// 输入shape: [b, s, 3, n, d]
// 输出shape: [b, s, 3, n, d]
template<typename T>
void invokeFusedRmsNormAndRope(T*           out,
                               const T*     input,
                               const T*     scale,
                               const T*     rope_emb,
                               float        eps,
                               int          batch_size,
                               int          seq_len,
                               int          head_num,
                               int          head_dim,
                               cudaStream_t stream)
{
    // 这里头维度乘2是为了只处理 qk 的数据
    dim3 grid(seq_len, head_num * 3, batch_size);
    dim3 block(std::min(head_dim / 2, 1024));
    fuseRmsNormAndRope<<<grid, block, 0, stream>>>(
        out, input, scale, rope_emb, eps, batch_size, seq_len, head_num, head_dim);
}

template<typename T>
void invokeFusedRmsNormCatAndRope(T*           out,
                                  const T*     hidden,
                                  const T*     encoder_hidden,
                                  const T*     hidden_rms_scale,
                                  const T*     encoder_rms_scale,
                                  const T*     rope_emb,
                                  float        eps,
                                  int          batch_size,
                                  int          hidden_seq_len,
                                  int          encoder_seq_len,
                                  int          head_num,
                                  int          head_dim,
                                  cudaStream_t stream)
{
    dim3 grid(hidden_seq_len + encoder_seq_len, head_num * 3, batch_size);
    dim3 block(std::min(head_dim / 2, 1024));
    fuseRmsNormCatAndRope<<<grid, block, 0, stream>>>(out,
                                                      hidden,
                                                      encoder_hidden,
                                                      hidden_rms_scale,
                                                      encoder_rms_scale,
                                                      rope_emb,
                                                      eps,
                                                      batch_size,
                                                      hidden_seq_len,
                                                      encoder_seq_len,
                                                      head_num,
                                                      head_dim);
}

template<typename T>
void invokeFusedBiasRmsNormAndRope(T*           out,
                                   const T*     input,
                                   const T*     bias,
                                   const T*     scale,
                                   const T*     rope_emb,
                                   float        eps,
                                   int          batch_size,
                                   int          seq_len,
                                   int          head_num,
                                   int          head_dim,
                                   cudaStream_t stream)
{
    dim3 grid(seq_len, head_num * 3, batch_size);
    dim3 block(std::min(head_dim / 2, 1024));
    fuseBiasRmsNormAndRope<<<grid, block, 0, stream>>>(
        out, input, bias, scale, rope_emb, eps, batch_size, seq_len, head_num, head_dim);
}

#define INSTANTIATE_INVOKE_FLUX_APPLY_ROPE(T)                                                                          \
    template void invokeFluxApplyRope(T*           out,                                                                \
                                      const T*     input,                                                              \
                                      const T*     rope_emb,                                                           \
                                      int          batch_size,                                                         \
                                      int          seq_len,                                                            \
                                      int          head_num,                                                           \
                                      int          head_dim,                                                           \
                                      cudaStream_t stream)

INSTANTIATE_INVOKE_FLUX_APPLY_ROPE(float);
INSTANTIATE_INVOKE_FLUX_APPLY_ROPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FLUX_APPLY_ROPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_FLUX_APPLY_ROPE

#define INSTANTIATE_INVOKE_FUSED_NORM_AND_ROPE(T)                                                                      \
    template void invokeFusedRmsNormAndRope(T*           out,                                                          \
                                            const T*     input,                                                        \
                                            const T*     scale,                                                        \
                                            const T*     rope_emb,                                                     \
                                            float        eps,                                                          \
                                            int          batch_size,                                                   \
                                            int          seq_len,                                                      \
                                            int          head_num,                                                     \
                                            int          head_dim,                                                     \
                                            cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_NORM_AND_ROPE(float);
INSTANTIATE_INVOKE_FUSED_NORM_AND_ROPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_NORM_AND_ROPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_FUSED_NORM_AND_ROPE

#define INSTANTIATE_INVOKE_FUSED_BIAS_NORM_AND_ROPE(T)                                                                 \
    template void invokeFusedBiasRmsNormAndRope(T*           out,                                                      \
                                                const T*     input,                                                    \
                                                const T*     bias,                                                     \
                                                const T*     scale,                                                    \
                                                const T*     rope_emb,                                                 \
                                                float        eps,                                                      \
                                                int          batch_size,                                               \
                                                int          seq_len,                                                  \
                                                int          head_num,                                                 \
                                                int          head_dim,                                                 \
                                                cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_BIAS_NORM_AND_ROPE(float);
INSTANTIATE_INVOKE_FUSED_BIAS_NORM_AND_ROPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_BIAS_NORM_AND_ROPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_FUSED_BIAS_NORM_AND_ROPE

#define INSTANTIATE_INVOKE_FUSED_NORM_CAT_AND_ROPE(T)                                                                  \
    template void invokeFusedRmsNormCatAndRope(T*           out,                                                       \
                                               const T*     hidden,                                                    \
                                               const T*     encoder_hidden,                                            \
                                               const T*     hidden_rms_scale,                                          \
                                               const T*     encoder_rms_scale,                                         \
                                               const T*     rope_emb,                                                  \
                                               float        eps,                                                       \
                                               int          batch_size,                                                \
                                               int          hidden_seq_len,                                            \
                                               int          encoder_seq_len,                                           \
                                               int          head_num,                                                  \
                                               int          head_dim,                                                  \
                                               cudaStream_t stream)

INSTANTIATE_INVOKE_FUSED_NORM_CAT_AND_ROPE(float);
INSTANTIATE_INVOKE_FUSED_NORM_CAT_AND_ROPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_FUSED_NORM_CAT_AND_ROPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_FUSED_NORM_CAT_AND_ROPE

}  // namespace lyradiff