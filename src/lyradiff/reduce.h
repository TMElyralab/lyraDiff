#pragma once
#include "common.h"

namespace lyradiff {
// 全位置 mask
#define FINAL_MASK 0xffffffff

__device__ __forceinline__ int getLaneId()
{
    int laneId;
    // 使用内联汇编操作，获取当前线程在 CUDA 线程块中的线程编号，并存到 laneId 变量中
    asm("mov.s32 %0, %laneid;" : "=r"(laneId));
    return laneId;
}

// 线程束间做广播
template<typename T>
__inline__ __device__ void warpBroadcast(T* val)
{
    *val = __shfl_sync(FINAL_MASK, *val, 0, 32);
}

// 线程束间做每个线程的累加和
template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

// block 间做每个线程的数据的累加和
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    return wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)0.0f) : 0.0f;
}

// __half2 数据的累加需要使用 __hadd2 接口，以下函数特例化 __half2 类型的 redcue 操作
__inline__ __device__ __half2 warpReduceSum(__half2 val)
{
    half2 tmp_val;
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp_val = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
        val     = __hadd2(tmp_val, val);
    }
    return val;
}

// __half2 类型的自生 2 个元素的累加
__inline__ __device__ __half __half2add(__half2 val)
{
    return __hadd(val.x, val.y);
}

__inline__ __device__ __half blockReduceSum(__half2 val)
{
    static __shared__ __half2 shared[32];
    int                       lane = threadIdx.x & 0x1f;
    int                       wid  = threadIdx.x >> 5;

    val = warpReduceSum<__half2>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    return (__half)(wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? (float)__half2add(shared[lane]) : 0.0f) :
                               0.0f);
}

// max 操作
template<typename T>
__inline__ __device__ T max_(T a, T b)
{
    return a > b ? a : b;
}

// 线程束之间的最大值
template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max_(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

// 线程块之间的最大值
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    return wid == 0 ? warpReduceMax(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)-1e20f) : (T)-1e20f;
}
}  // namespace lyradiff