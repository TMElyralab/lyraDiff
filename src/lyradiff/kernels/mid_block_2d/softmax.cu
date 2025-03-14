#include "softmax.h"
#include "src/lyradiff/kernels/reduce_kernel_utils.cuh"
// #include "src/lyradiff/reduce.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
inline int divUp(int a, int n)
{
    return (a + n - 1) / n;
}

inline __device__ void dprint(int x)
{
    if (blockIdx.x == blockIdx.y && blockIdx.z == blockIdx.y && blockIdx.z == 0) {
        if (threadIdx.x == 0) {
            printf("debug %d %d %d\n", x, x, x);
        }
    }
}

// std::string getCudaErrorAsString()
// {
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         // 返回错误描述的字符串
//         return std::string(cudaGetErrorString(error));
//     }
//     // 同步设备，以等待内核完成并捕获任何错误
//     error = cudaDeviceSynchronize();
//     if (error != cudaSuccess) {
//         // 返回错误描述的字符串
//         return std::string(cudaGetErrorString(error));
//     }
//     return "No error";  // 没有错误发生
// }

template<typename T, typename T_IN, int ITEMS_PER_THREAD>
__global__ void __launch_bounds__(1024) softmax_kernel(T*          attn_score,
                                                       const T_IN* qk,
                                                       const T*    attn_mask,
                                                       const T*    linear_bias_slopes,
                                                       const int   batch_size,
                                                       const int   head_num,
                                                       const int   q_length,
                                                       const int   k_length,
                                                       const float qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    const int64_t bi = blockIdx.y;  // Batch index.
    const int64_t hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float)linear_bias_slopes[hi] : 0.0f;

    // Loop along with Q dimension.
    for (int64_t qi = blockIdx.x; qi < q_length; qi += gridDim.x) {

        float   data[ITEMS_PER_THREAD];
        int64_t qk_offset;
        float   local_max = -1e20f;

        // Loop along with K dimension.
        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            int64_t ki = blockDim.x * i + threadIdx.x;  // Index of K dimension.
            qk_offset  = ((bi * head_num + hi) * q_length + qi) * k_length + ki;

            float qk_val  = static_cast<float>(qk[qk_offset]);
            float qk_bias = 0.0f;

            if (linear_bias_slopes != nullptr) {
                // We don't handle the upper diagonal (ki > qi) separately, whose values
                // are negligible due to the negative infinity mask. And it matches with
                // the HF's implementation.
                qk_bias += static_cast<float>(linear_bias_slope * (ki - qi));
            }

            int64_t mask_offset = (bi * q_length + qi) * k_length + ki;

            if (attn_mask != nullptr) {
                float mask_val = static_cast<float>(ldg(&attn_mask[mask_offset]));
                qk_bias += (1.0f - mask_val) * -10000.0f;
            }

            data[i]   = qk_scale * qk_val + qk_bias;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            qk_offset             = ((bi * head_num + hi) * q_length + qi) * k_length + blockDim.x * i + threadIdx.x;
            attn_score[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<typename T, int ITEMS_PER_THREAD>
__global__ void __launch_bounds__(1024) softmax_kernel_h2(T*        attn_score,
                                                          const T*  qk_buf,
                                                          const T*  attn_mask,
                                                          const T*  linear_bias_slopes,
                                                          const int batch_size,
                                                          const int head_num,
                                                          const int q_length,
                                                          const int k_length,
                                                          const T   qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = nullptr;

    if (attn_mask != nullptr) {
        attn_mask_h2 = reinterpret_cast<const T2*>(attn_mask);
    }

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale_h2 = cuda_cast<T2>(qk_scale);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {
        T2    data[ITEMS_PER_THREAD];
        int   qk_offset;
        float local_max = -1e20f;

        // Loop over k dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki          = blockDim.x * i + threadIdx.x;
            qk_offset       = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + ki;
            int mask_offset = (bi * q_length + qi) * (k_length / 2) + ki;

            // The value of QK^T matrix at (qi, ki).
            T2 qk = qk_buf_h2[qk_offset];
            // The bias value to the position (qi, ki) including both mask and positional bias.
            T2 qk_bias = ZERO;

            if (linear_bias_slopes != nullptr) {
                // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                // separately, whose values are negligible due to the negative infinity mask.
                T2 dist(2.0f * ki - qi, 2.0f * ki + 1 - qi);
                qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(linear_bias_slope, dist));
            }
            if (attn_mask_h2 != nullptr) {
                T2 mask_val = ldg(&attn_mask_h2[mask_offset]);
                qk_bias     = hadd2<T2>(qk_bias, hmul2<T2>(hsub2<T2>(ONE, mask_val), NEG_INFTY));
            }

            data[i]   = hadd2<T2>(hmul2<T2>(qk, qk_scale_h2), qk_bias);
            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + blockDim.x * i + threadIdx.x;
            attn_score_h2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

template<typename T, int K_ITEMS_PER_THREAD, int Q_ITEMS_PER_THREAD>
__global__ void __launch_bounds__(1024) softmax_kernel_h2_v2(T*        attn_score,
                                                             const T*  qk_buf,
                                                             const T*  attn_mask,
                                                             const T*  linear_bias_slopes,
                                                             const int batch_size,
                                                             const int head_num,
                                                             const int q_length,
                                                             const int k_length,
                                                             const T   scalar)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    // QK^T matrix of shape (batch_size, head_num, q_length, k_length / 2)
    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = nullptr;

    if (attn_mask != nullptr) {
        attn_mask_h2 = reinterpret_cast<const T2*>(attn_mask);
    }

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale = cuda_cast<T2>(scalar);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    __shared__ float s_sum[Q_ITEMS_PER_THREAD], s_max[Q_ITEMS_PER_THREAD];

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x * Q_ITEMS_PER_THREAD) {
        T2 data[Q_ITEMS_PER_THREAD][K_ITEMS_PER_THREAD];

        int qk_offset[Q_ITEMS_PER_THREAD];

        float local_max[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_max[j] = -1e20f;
        }

        // Loop over k dimension.
        const int Q_ITEMS = min((q_length - qi + gridDim.x - 1) / gridDim.x, Q_ITEMS_PER_THREAD);
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki = blockDim.x * i + threadIdx.x;

            int mask_offset[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j]   = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
                mask_offset[j] = (bi * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
            }

            T2 mask_val[Q_ITEMS_PER_THREAD];
            if (attn_mask_h2 != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    mask_val[j] = ldg(&attn_mask_h2[mask_offset[j]]);
                }
            }

            T2 qk[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk[j] = qk_buf_h2[qk_offset[j]];
            }

            T2 pos_bias[Q_ITEMS_PER_THREAD];
            if (linear_bias_slopes != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                    // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                    // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                    // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                    // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                    // separately, whose values are negligible due to the negative infinity mask.
                    int qidx = qi + j * gridDim.x;
                    T2  dist(2.0f * ki - qidx, 2.0f * ki + 1 - qidx);
                    pos_bias[j] = hmul2<T2>(linear_bias_slope, dist);
                }
            }
            if (attn_mask_h2 != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    mask_val[j] = hmul2<T2>(hsub2<T2>(ONE, mask_val[j]), NEG_INFTY);
                }
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                T2 val = hmul2<T2>(qk_scale, qk[j]);
                if (attn_mask_h2 != nullptr) {
                    val = hadd2<T2>(val, mask_val[j]);
                }
                if (linear_bias_slopes != nullptr) {
                    val = hadd2<T2>(val, pos_bias[j]);
                }
                data[j][i]   = val;
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }
        else {
            blockReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; ++j) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }
        else {
            blockReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + blockDim.x * i
                               + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                attn_score_h2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

#define LAUNCH_MASKED_SOFTMAX_(T_, ITEMS_PER_THREAD)                                                                   \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    block.x = divUp(block.x, 32) * 32;                                                                                 \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_h2_v2<T_, ITEMS_PER_THREAD, 4>                                                              \
                <<<grid, block, 0, stream>>>((T_*)param.attention_score,                                               \
                                             (const T_*)param.qk,                                                      \
                                             (const T_*)param.attention_mask,                                          \
                                             (const T_*)param.linear_bias_slopes,                                      \
                                             param.batch_size,                                                         \
                                             param.num_heads,                                                          \
                                             param.q_length,                                                           \
                                             param.k_length,                                                           \
                                             (const T_)param.qk_scale);                                                \
        }                                                                                                              \
        else {                                                                                                         \
            softmax_kernel_h2<T_, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((T_*)param.attention_score,            \
                                                                                (const T_*)param.qk,                   \
                                                                                (const T_*)param.attention_mask,       \
                                                                                (const T_*)param.linear_bias_slopes,   \
                                                                                param.batch_size,                      \
                                                                                param.num_heads,                       \
                                                                                param.q_length,                        \
                                                                                param.k_length,                        \
                                                                                (const T_)param.qk_scale);             \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        \                                                            
        softmax_kernel<T, T_IN, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(param.attention_score,                   \
                                                                              param.qk,                                \
                                                                              param.attention_mask,                    \
                                                                              param.linear_bias_slopes,                \
                                                                              param.batch_size,                        \
                                                                              param.num_heads,                         \
                                                                              param.q_length,                          \
                                                                              param.k_length,                          \
                                                                              param.qk_scale);                         \
    }

#define LAUNCH_MASKED_SOFTMAX(ITEMS_PER_THREAD) LAUNCH_MASKED_SOFTMAX_(half, ITEMS_PER_THREAD)

template<typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    // std::cout << "block.x " << block.x << std::endl;
    // cudaDeviceSynchronize();
    if (block.x > 32768) {
        LYRA_CHECK("error");  // Not implemented - it's not clear we want to use the unfused kernel in that case.
    }
    else if (block.x > 16384) {
        LAUNCH_MASKED_SOFTMAX(32)
    }
    else if (block.x > 8192) {
        LAUNCH_MASKED_SOFTMAX(16)
    }
    else if (block.x > 4096) {
        LAUNCH_MASKED_SOFTMAX(8)
    }
    else if (block.x > 2048) {
        LAUNCH_MASKED_SOFTMAX(4)
    }
    else if (block.x > 1024) {
        LAUNCH_MASKED_SOFTMAX(2)
    }
    else if (block.x > 0) {
        LAUNCH_MASKED_SOFTMAX(1)
    }
    // cudaDeviceSynchronize();
    // std::cout << "last cuda error: " << getCudaErrorAsString() << std::endl;
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, half>& param, cudaStream_t stream);

int log2_ceil(int value)
{
    int log2_value = 0;
    while ((1 << log2_value) < value)
        ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
    __device__ __forceinline__ T operator()(T a, T b) const
    {
        return a + b;
    }
};

template<typename T>
struct Max {
    __device__ __forceinline__ T operator()(T a, T b) const
    {
        return a < b ? b : a;
    }
};

template<typename T>
__device__ __forceinline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template<typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum)
{
    ReduceOp<acc_t> r;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i]  = r(sum[i], b);
        }
    }
}

/*
 * Extended softmax (from native aten pytorch) with following additional features
 * 1) input scaling
 * 2) Explicit masking
 */
template<typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_softmax_warp_forward(
    output_t* dst, const input_t* src, const acc_t scale, int micro_batch_size, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    // warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two    = 1 << log2_elements;
    constexpr int WARP_SIZE            = (next_power_of_two < 32) ? next_power_of_two : 32;
    constexpr int WARP_ITERATIONS      = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH           = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, )
    // gridDim/blockIdx = (seq_len, attn_heads, batches)
    int first_batch =
        (blockDim.y * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) + threadIdx.y) * WARP_BATCH;

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
    dst += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;

    // load data from global memory
    acc_t   elements[WARP_BATCH][WARP_ITERATIONS];
    input_t temp_data[ELEMENTS_PER_LDG_STG];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;

#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

            if (element_index < batch_element_count) {
                int itr_idx = i * element_count + it * WARP_SIZE;
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_data, src + itr_idx);

#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    elements[i][it + element] = (acc_t)temp_data[element] * scale;
                }
            }
            else {
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
                }
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        max_value[i] = elements[i][0];
#pragma unroll
        for (int it = 1; it < WARP_ITERATIONS; ++it) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    // compute scale value to account for full mask
    acc_t scale_value[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        scale_value[i] = (max_value[i] == -10000.0) ? 0.0 : 1.0;
    }

    acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            elements[i][it] = std::exp((elements[i][it] - max_value[i]));
            sum[i] += elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        if (i >= local_batches)
            break;
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    out[element] = elements[i][it + element] * scale_value[i] / sum[i];
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count + it * WARP_SIZE, out);
            }
            else {
                break;
            }
        }
    }
}

template<typename input_t, typename output_t>
void dispatch_scaled_softmax_forward(output_t*      dst,
                                     const input_t* src,
                                     const input_t  scale,
                                     int            query_seq_len,
                                     int            key_seq_len,
                                     int            batches,
                                     int            attn_heads,
                                     cudaStream_t   stream)
{
    // TORCH_INTERNAL_ASSERT(key_seq_len >= 0 && key_seq_len <= 8192);
    if (key_seq_len == 0) {
        return;
    }
    else {
        int       log2_elements     = log2_ceil(key_seq_len);
        const int next_power_of_two = 1 << log2_elements;
        int       batch_count       = batches * attn_heads * query_seq_len;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int  warps_per_block   = (threads_per_block / warp_size);
        int  batches_per_block = warps_per_block * batches_per_warp;
        dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0:  // 1
                scaled_softmax_warp_forward<input_t, output_t, float, 0>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 1:  // 2
                scaled_softmax_warp_forward<input_t, output_t, float, 1>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 2:  // 4
                scaled_softmax_warp_forward<input_t, output_t, float, 2>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 3:  // 8
                scaled_softmax_warp_forward<input_t, output_t, float, 3>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 4:  // 16
                scaled_softmax_warp_forward<input_t, output_t, float, 4>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 5:  // 32
                scaled_softmax_warp_forward<input_t, output_t, float, 5>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 6:  // 64
                scaled_softmax_warp_forward<input_t, output_t, float, 6>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 7:  // 128
                scaled_softmax_warp_forward<input_t, output_t, float, 7>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 8:  // 256
                scaled_softmax_warp_forward<input_t, output_t, float, 8>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 9:  // 512
                scaled_softmax_warp_forward<input_t, output_t, float, 9>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 10:  // 1024
                scaled_softmax_warp_forward<input_t, output_t, float, 10>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 11:  // 2048
                scaled_softmax_warp_forward<input_t, output_t, float, 11>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 12:  // 4096
                scaled_softmax_warp_forward<input_t, output_t, float, 12>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 13:  // 8192
                scaled_softmax_warp_forward<input_t, output_t, float, 13>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            case 14:  // 16384
                scaled_softmax_warp_forward<input_t, output_t, float, 14>
                    <<<blocks, threads, 0, stream>>>(dst, src, scale, batch_count, key_seq_len);
                break;
            default:
                break;
        }
    }
}

#define INSTANTIATE_DISPATCH_SCALED_SOFTMAX_FORWARD(input_t, output_t)          \
    template void dispatch_scaled_softmax_forward(output_t*      dst,           \
                                              const input_t* src,               \ 
                                              const input_t  scale,             \ 
                                              int            query_seq_len,     \
                                              int            key_seq_len,       \
                                              int            batches,           \
                                              int            attn_heads,        \
                                              cudaStream_t   stream)

INSTANTIATE_DISPATCH_SCALED_SOFTMAX_FORWARD(float, float);
INSTANTIATE_DISPATCH_SCALED_SOFTMAX_FORWARD(half, half);
#undef INSTANTIATE_DISPATCH_SCALED_SOFTMAX_FORWARD

}  // namespace lyradiff