#include "layer_norm.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include <stdio.h>
namespace lyradiff {

constexpr int kWarpSize = 32;

template<typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count)
{
    // Use Welford Online algorithem to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    *count += 1;
    T delta1 = val - *mean;
    *mean += delta1 / *count;
    T delta2 = val - *mean;
    *m2 += delta1 * delta2;
}

template<typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count)
{
    if (b_count == cuda_cast<T>(0.0)) {
        return;
    }
    T new_count = *count + b_count;
    T nb_over_n = b_count / new_count;
    T delta     = b_mean - *mean;
    *mean += delta * nb_over_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count)
{
    *mean  = thread_mean;
    *m2    = thread_m2;
    *count = thread_count;
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        T b_mean  = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
        T b_m2    = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
        T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
        WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

template<typename T>
__inline__ __device__ void
WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count, T* result_mean, T* result_m2, T* result_count)
{
    __shared__ T mean_shared[kWarpSize];
    __shared__ T m2_shared[kWarpSize];
    __shared__ T count_shared[kWarpSize];
    __shared__ T mean_result_broadcast;
    __shared__ T m2_result_broadcast;
    __shared__ T count_result_broadcast;
    const int    lid        = threadIdx.x % kWarpSize;
    const int    wid        = threadIdx.x / kWarpSize;
    T            warp_mean  = 0;
    T            warp_m2    = 0;
    T            warp_count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
    __syncthreads();
    if (lid == 0) {
        mean_shared[wid]  = warp_mean;
        m2_shared[wid]    = warp_m2;
        count_shared[wid] = warp_count;
    }
    __syncthreads();
    if (wid == 0) {
        if (threadIdx.x < blockDim.x / kWarpSize) {
            warp_mean  = mean_shared[lid];
            warp_m2    = m2_shared[lid];
            warp_count = count_shared[lid];
        }
        else {
            warp_mean  = static_cast<T>(0);
            warp_m2    = static_cast<T>(0);
            warp_count = static_cast<T>(0);
        }
        __syncwarp();
        T block_mean  = 0;
        T block_m2    = 0;
        T block_count = 0;
        WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
        if (lid == 0) {
            mean_result_broadcast  = block_mean;
            m2_result_broadcast    = block_m2;
            count_result_broadcast = block_count;
        }
    }
    __syncthreads();
    *result_mean  = mean_result_broadcast;
    *result_m2    = m2_result_broadcast;
    *result_count = count_result_broadcast;
}

template<typename T, int32_t cols_per_thread>
__global__ void LayerNormBlockSMemImpl(
    T* dst, const T* src, const T* gamma, const T* beta, const double epsilon, const int32_t rows, const int32_t cols)
{
    // extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    // auto*                                                     buf = reinterpret_cast<T*>(shared_buf);

    int       row_id = blockIdx.x;
    const int tid    = threadIdx.x;

    constexpr int32_t num_iters_per_thread = cols_per_thread / 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2
    __align__(2 * sizeof(T)) T2 buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 gamma_buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 beta_buf[num_iters_per_thread];
    T                           thread_mean  = 0.0;
    T                           thread_m2    = 0.0;
    T                           delta        = 0.0;
    T                           delta2       = 0.0;
    T                           thread_count = 0.0;

    // 加载每个线程的数据到寄存器
#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        T2      ele2   = *reinterpret_cast<const T2*>(src + offset);
        // 第一个元素 welford
        thread_count += 1;
        delta = ele2.x - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.x - thread_mean;
        thread_m2 += delta * delta2;

        // 第二个元素 welford
        thread_count += 1;
        delta = ele2.y - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.y - thread_mean;
        thread_m2 += delta * delta2;

        buf[i] = ele2;

        int32_t gb_offset = tid * cols_per_thread + i * 2;
        gamma_buf[i]      = *reinterpret_cast<const T2*>(gamma + gb_offset);
        beta_buf[i]       = *reinterpret_cast<const T2*>(beta + gb_offset);
    }

    // 线程束之间每个线程使用 welford 算法聚合 mean, m2, 以及元素数量 count

    T row_mean  = 0;
    T row_m2    = 0;
    T row_count = 0;

    WelfordBlockAllReduce<T>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);

    // T row_mean     = warp_mean;
    T row_variance = cuda_max<T>(row_m2 / static_cast<T>(row_count), static_cast<T>(0.0));
    T row_inv_var  = rsqrt(row_variance + static_cast<T>(epsilon));

#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        // 寄存器上数据算数操作
        T2 ele2 = buf[i];
        T2 g2   = gamma_buf[i];
        T2 b2   = beta_buf[i];

        ele2.x = (ele2.x - row_mean) * row_inv_var;
        ele2.y = (ele2.y - row_mean) * row_inv_var;

        ele2.x = g2.x * ele2.x + b2.x;
        ele2.y = g2.y * ele2.y + b2.y;

        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        // 会写全局显存变量
        *reinterpret_cast<T2*>(&dst[offset]) = ele2;
    }
}


template<typename T, int32_t cols_per_thread>
__global__ void warpLayerNorm(
    T* dst, const T* src, const T* gamma, const T* beta, const double epsilon, const int32_t rows, const int32_t cols)
{
    // 一个线程束，最大分配 32 个线程
    // 一个线程在 A100 上可最大获得 255 bytes 的寄存器:
    //  分配 32 个线程， 每个线程最大可获得 255bytes 寄存器，half2 类型，每个线程一次处理 2 个元素，4个字节
    // 当 cols = 320 时：
    //      需迭代 320/32/2=5 次，寄存器 5*4 = 20 个bytes
    // 当 cols = 640 时：
    //      需迭代 640/32/2=10 次，寄存器 10*4 = 40 个bytes
    // 当 cols = 1024 时：
    //      需迭代 1024/32/2=16 次，寄存器 16*4 = 64 个bytes
    // 当 cols = 1280 时：
    //      需迭代 1280/32/2=20 次，寄存器 20*4 = 80 个bytes  （但我不确定目前这个尺寸是warp上做更快还是 block 上更快）

    // 数据布局：[B, C, NHiddens] ---> [B*C, NHiddens] ---> [Rows, Cols] ---> Flatten
    // gamma, beta: [NHiddens]
    // Grid: [B, C]
    // Block: [32,]
    constexpr int32_t num_iters_per_thread = cols_per_thread / 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2
    __align__(2 * sizeof(T)) T2 buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 gamma_buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 beta_buf[num_iters_per_thread];
    T                           thread_mean  = 0.0;
    T                           thread_m2    = 0.0;
    T                           delta        = 0.0;
    T                           delta2       = 0.0;
    T                           thread_count = 0.0;

    // int row_id = blockIdx.x * gridDim.y + blockIdx.y;
    int row_id = blockIdx.x;

    int tid = threadIdx.x;

// 加载每个线程的数据到寄存器
#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        T2      ele2   = *reinterpret_cast<const T2*>(src + offset);
        // 第一个元素 welford
        thread_count += 1;
        delta = ele2.x - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.x - thread_mean;
        thread_m2 += delta * delta2;

        // 第二个元素 welford
        thread_count += 1;
        delta = ele2.y - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.y - thread_mean;
        thread_m2 += delta * delta2;

        buf[i] = ele2;

        int32_t gb_offset = tid * cols_per_thread + i * 2;
        gamma_buf[i]      = *reinterpret_cast<const T2*>(gamma + gb_offset);
        beta_buf[i]       = *reinterpret_cast<const T2*>(beta + gb_offset);
    }

    // 线程束之间每个线程使用 welford 算法聚合 mean, m2, 以及元素数量 count
    T warp_mean, warp_m2;
    T warp_count;
    warp_mean  = thread_mean;
    warp_m2    = thread_m2;
    warp_count = thread_count;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        T b_mean  = __shfl_down_sync(0xffffffff, warp_mean, mask, 32);
        T b_m2    = __shfl_down_sync(0xffffffff, warp_m2, mask, 32);
        T b_count = __shfl_down_sync(0xffffffff, warp_count, mask, 32);

        T new_count = warp_count + b_count;
        T nb_over_n = b_count / new_count;
        T delta     = b_mean - warp_mean;
        warp_mean += delta * nb_over_n;
        warp_m2 += b_m2 + delta * delta * warp_count * nb_over_n;
        warp_count = new_count;
    }

    // welford 聚合求得的值广播到每个线程
    warp_mean  = __shfl_sync(0xffffffff, warp_mean, 0, 32);
    warp_m2    = __shfl_sync(0xffffffff, warp_m2, 0, 32);
    warp_count = __shfl_sync(0xffffffff, warp_count, 0, 32);

    T row_mean     = warp_mean;
    T row_variance = cuda_max<T>(warp_m2 / static_cast<T>(warp_count), static_cast<T>(0.0));
    T row_inv_var  = rsqrt(row_variance + static_cast<T>(epsilon));

#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        // 寄存器上数据算数操作
        T2 ele2 = buf[i];
        T2 g2   = gamma_buf[i];
        T2 b2   = beta_buf[i];

        ele2.x = (ele2.x - row_mean) * row_inv_var;
        ele2.y = (ele2.y - row_mean) * row_inv_var;

        ele2.x = g2.x * ele2.x + b2.x;
        ele2.y = g2.y * ele2.y + b2.y;

        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        // 会写全局显存变量
        *reinterpret_cast<T2*>(&dst[offset]) = ele2;
    }
}

template<typename T, int32_t cols_per_thread>
__global__ void warpLayerNormWithShiftAndScale(
    T* dst, const T* src, const T* scale, const T* shift, const double epsilon, const int32_t rows, const int32_t cols)
{
    // 一个线程束，最大分配 32 个线程
    // 一个线程在 A100 上可最大获得 255 bytes 的寄存器:
    //  分配 32 个线程， 每个线程最大可获得 255bytes 寄存器，half2 类型，每个线程一次处理 2 个元素，4个字节
    // 当 cols = 320 时：
    //      需迭代 320/32/2=5 次，寄存器 5*4 = 20 个bytes
    // 当 cols = 640 时：
    //      需迭代 640/32/2=10 次，寄存器 10*4 = 40 个bytes
    // 当 cols = 1024 时：
    //      需迭代 1024/32/2=16 次，寄存器 16*4 = 64 个bytes
    // 当 cols = 1280 时：
    //      需迭代 1280/32/2=20 次，寄存器 20*4 = 80 个bytes  （但我不确定目前这个尺寸是warp上做更快还是 block 上更快）

    // 数据布局：[B, C, NHiddens] ---> [B*C, NHiddens] ---> [Rows, Cols] ---> Flatten
    // scale, shift: [B, NHiddens]
    // Grid: [B, C]
    // Block: [32,]
    constexpr int32_t num_iters_per_thread = cols_per_thread / 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2
    __align__(2 * sizeof(T)) T2 buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 shift_buf[num_iters_per_thread];
    __align__(2 * sizeof(T)) T2 scale_buf[num_iters_per_thread];
    T                           thread_mean  = 0.0;
    T                           thread_m2    = 0.0;
    T                           delta        = 0.0;
    T                           delta2       = 0.0;
    T                           thread_count = 0.0;
    T                           one          = 1.0;

    // int row_id = blockIdx.x * gridDim.y + blockIdx.y;
    int row_id = blockIdx.x + blockIdx.y * rows;

    int tid = threadIdx.x;

// 加载每个线程的数据到寄存器
#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        T2      ele2   = *reinterpret_cast<const T2*>(src + offset);
        // 第一个元素 welford
        thread_count += 1;
        delta = ele2.x - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.x - thread_mean;
        thread_m2 += delta * delta2;

        // 第二个元素 welford
        thread_count += 1;
        delta = ele2.y - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.y - thread_mean;
        thread_m2 += delta * delta2;

        buf[i] = ele2;

        int32_t gb_offset = blockIdx.y * cols + tid * cols_per_thread + i * 2;
        scale_buf[i]      = *reinterpret_cast<const T2*>(scale + gb_offset);
        shift_buf[i]      = *reinterpret_cast<const T2*>(shift + gb_offset);
    }

    // 线程束之间每个线程使用 welford 算法聚合 mean, m2, 以及元素数量 count
    T warp_mean, warp_m2;
    T warp_count;
    warp_mean  = thread_mean;
    warp_m2    = thread_m2;
    warp_count = thread_count;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        T b_mean  = __shfl_down_sync(0xffffffff, warp_mean, mask, 32);
        T b_m2    = __shfl_down_sync(0xffffffff, warp_m2, mask, 32);
        T b_count = __shfl_down_sync(0xffffffff, warp_count, mask, 32);

        T new_count = warp_count + b_count;
        T nb_over_n = b_count / new_count;
        T delta     = b_mean - warp_mean;
        warp_mean += delta * nb_over_n;
        warp_m2 += b_m2 + delta * delta * warp_count * nb_over_n;
        warp_count = new_count;
    }

    // welford 聚合求得的值广播到每个线程
    warp_mean  = __shfl_sync(0xffffffff, warp_mean, 0, 32);
    warp_m2    = __shfl_sync(0xffffffff, warp_m2, 0, 32);
    warp_count = __shfl_sync(0xffffffff, warp_count, 0, 32);

    T row_mean     = warp_mean;
    T row_variance = cuda_max<T>(warp_m2 / static_cast<T>(warp_count), static_cast<T>(0.0));
    T row_inv_var  = rsqrt(row_variance + static_cast<T>(epsilon));

#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        // 寄存器上数据算数操作
        T2 ele2   = buf[i];
        T2 scale2 = scale_buf[i];
        T2 shift2 = shift_buf[i];

        ele2.x = (ele2.x - row_mean) * row_inv_var;
        ele2.y = (ele2.y - row_mean) * row_inv_var;

        ele2.x = (scale2.x + one) * ele2.x + shift2.x;
        ele2.y = (scale2.y + one) * ele2.y + shift2.y;

        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        // 会写全局显存变量
        *reinterpret_cast<T2*>(&dst[offset]) = ele2;
    }
}

template<typename T, int32_t cols_per_thread>
__global__ void LayerNormBlockWithShiftAndScaleSMemImpl(
    T* dst, const T* src, const T* scale, const T* shift, const double epsilon, const int32_t rows, const int32_t cols)
{
    // extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    // auto*                                                     buf = reinterpret_cast<T*>(shared_buf);
    // scale, shift: [B, NHiddens]
    int row_id = blockIdx.x + blockIdx.y * rows;
    int tid    = threadIdx.x;

    constexpr int32_t num_iters_per_thread = cols_per_thread / 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2
    __align__(2 * sizeof(float)) float2 buf[num_iters_per_thread];
    __align__(2 * sizeof(float)) T2     shift_buf[num_iters_per_thread];
    __align__(2 * sizeof(float)) T2     scale_buf[num_iters_per_thread];
    float                               thread_mean  = 0.0;
    float                               thread_m2    = 0.0;
    float                               delta        = 0.0;
    float                               delta2       = 0.0;
    float                               thread_count = 0.0;
    T                                   one          = 1.0;
    // 加载每个线程的数据到寄存器
#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        T2      tmp2   = *reinterpret_cast<const T2*>(src + offset);
        float2  ele2   = cuda_cast<float2>(tmp2);
        // 第一个元素 welford
        thread_count += 1;
        delta = ele2.x - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.x - thread_mean;
        thread_m2 += delta * delta2;

        // 第二个元素 welford
        thread_count += 1;
        delta = ele2.y - thread_mean;
        thread_mean += delta / thread_count;
        delta2 = ele2.y - thread_mean;
        thread_m2 += delta * delta2;

        buf[i] = ele2;

        int32_t gb_offset = blockIdx.y * cols + tid * cols_per_thread + i * 2;
        scale_buf[i]      = *reinterpret_cast<const T2*>(scale + gb_offset);
        shift_buf[i]      = *reinterpret_cast<const T2*>(shift + gb_offset);
    }

    // 线程束之间每个线程使用 welford 算法聚合 mean, m2, 以及元素数量 count

    float row_mean  = 0;
    float row_m2    = 0;
    float row_count = 0;

    WelfordBlockAllReduce<float>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);

    // T row_mean     = warp_mean;
    float row_variance = cuda_max<float>(row_m2 / static_cast<float>(row_count), static_cast<float>(0.0));
    float row_inv_var  = rsqrt(row_variance + static_cast<float>(epsilon));

#pragma unroll
    for (int32_t i = 0; i < num_iters_per_thread; ++i) {
        // 寄存器上数据算数操作
        float2 ele2   = buf[i];
        T2     scale2 = scale_buf[i];
        T2     shift2 = shift_buf[i];

        ele2.x  = (ele2.x - row_mean) * row_inv_var;
        ele2.y  = (ele2.y - row_mean) * row_inv_var;
        T2 tmp2 = cuda_cast<T2>(ele2);

        tmp2.x = (scale2.x + one) * tmp2.x + shift2.x;
        tmp2.y = (scale2.y + one) * tmp2.y + shift2.y;

        int32_t offset = row_id * cols + tid * cols_per_thread + i * 2;
        // 会写全局显存变量
        *reinterpret_cast<T2*>(&dst[offset]) = tmp2;
    }
}

template<typename T>
void invokeLayerNorm(T*           dst,
                     const T*     src,
                     const T*     gamma,
                     const T*     beta,
                     size_t       batch_size,
                     size_t       channels,
                     size_t       nhiddens,
                     cudaStream_t stream,
                     const double eps)
{
    const int32_t rows = batch_size * channels;
    const int32_t cols = nhiddens;

    // dim3 grid(batch_size, channels);
    // printf("layernorm nhiddens %d\n", nhiddens);

    dim3 grid(rows);

    dim3 block(32);
    switch (cols) {
        case 64 /* constant-expression */:
            /* code */
            warpLayerNorm<T, 2><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        case 320 /* constant-expression */:
            /* code */
            warpLayerNorm<T, 10><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        case 640:
            warpLayerNorm<T, 20><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        case 1024:
            warpLayerNorm<T, 32><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        case 1280:  // 1280 目前使用 LayerNormBlockSMemImpl
            // warpLayerNorm<T, 40><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            LayerNormBlockSMemImpl<T, 10><<<grid, 128, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        case 768:
            warpLayerNorm<T, 24><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            break;

        default:
            break;
    }
}

template<typename T>
void invokeLayerNormWithShiftAndScale(T*           dst,
                                      const T*     src,
                                      const T*     scale,
                                      const T*     shift,
                                      size_t       batch_size,
                                      size_t       channels,
                                      size_t       nhiddens,
                                      cudaStream_t stream,
                                      const double eps)
{
    const int32_t rows = channels;
    const int32_t cols = nhiddens;

    // dim3 grid(batch_size, channels);

    dim3 grid(rows, batch_size);

    dim3 block(32);
    switch (cols) {
        case 64 /* constant-expression */:
            /* code */
            warpLayerNormWithShiftAndScale<T, 2><<<grid, block, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;

        case 320 /* constant-expression */:
            /* code */
            warpLayerNormWithShiftAndScale<T, 10><<<grid, block, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;

        case 640:
            warpLayerNormWithShiftAndScale<T, 20><<<grid, block, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;

        case 1024:
            warpLayerNormWithShiftAndScale<T, 32><<<grid, block, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;

        case 1280:  // 1280 目前使用 LayerNormBlockSMemImpl
            // warpLayerNorm<T, 40><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            LayerNormBlockWithShiftAndScaleSMemImpl<T, 10>
                <<<grid, 128, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;
        case 3072:  // 3072 目前使用 LayerNormBlockSMemImpl
            // warpLayerNorm<T, 40><<<grid, block, 0, stream>>>(dst, src, gamma, beta, eps, rows, cols);
            LayerNormBlockWithShiftAndScaleSMemImpl<T, 24>
                <<<grid, 128, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;
        case 768:
            warpLayerNormWithShiftAndScale<T, 24><<<grid, block, 0, stream>>>(dst, src, scale, shift, eps, rows, cols);
            break;

        default:
            break;
    }
}

// 为 float 和 half 做模板特化 (BCN)
#define INSTANTIATE_INVOKE_LAYER_NORM(T)                                                                               \
    template void invokeLayerNorm(T*           dst,                                                                    \
                                  const T*     src,                                                                    \
                                  const T*     gamma,                                                                  \
                                  const T*     beta,                                                                   \
                                  size_t       batch_size,                                                             \
                                  size_t       channels,                                                               \
                                  size_t       nhiddens,                                                               \
                                  cudaStream_t stream,                                                                 \
                                  const double eps)

INSTANTIATE_INVOKE_LAYER_NORM(float);
INSTANTIATE_INVOKE_LAYER_NORM(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_LAYER_NORM(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_LAYER_NORM

// 为 float 和 half 做模板特化 (BCN)
#define INSTANTIATE_INVOKE_LAYER_NORM_WITH_SHIFT_AND_SCALE(T)                                                          \
    template void invokeLayerNormWithShiftAndScale(T*           dst,                                                   \
                                                   const T*     src,                                                   \
                                                   const T*     scale,                                                 \
                                                   const T*     shift,                                                 \
                                                   size_t       batch_size,                                            \
                                                   size_t       channels,                                              \
                                                   size_t       nhiddens,                                              \
                                                   cudaStream_t stream,                                                \
                                                   const double eps)

INSTANTIATE_INVOKE_LAYER_NORM_WITH_SHIFT_AND_SCALE(float);
INSTANTIATE_INVOKE_LAYER_NORM_WITH_SHIFT_AND_SCALE(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_LAYER_NORM_WITH_SHIFT_AND_SCALE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_LAYER_NORM_WITH_SHIFT_AND_SCALE

}  // namespace lyradiff