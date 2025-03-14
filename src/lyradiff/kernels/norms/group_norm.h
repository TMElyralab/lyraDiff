#pragma once

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

struct GroupSums {
    // 是否为一个 group 中的第一个元素，1 表示是，0表示否
    int32_t flag;
    // 累计和
    float sum;
    // 累计和的平方
    float sumSq;
};

struct GroupSumsOp {
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b)
    {
        GroupSums dst;
        dst.sum   = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag  = a.flag + b.flag;
        return dst;
    }
};

static inline __device__ __host__ float sigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

static inline int32_t divUp(int32_t m, int32_t n)
{
    return (m + n - 1) / n;
}

template<typename T>
void invokeGroupNorm(T*           dst,
                     const T*     input,
                     const T*     gamma,
                     const T*     beta,
                     double*      caches,
                     const size_t batch_size,
                     const size_t height,
                     const size_t width,
                     const size_t num_channels,
                     const size_t num_groups,
                     const bool   use_swish,
                     cudaStream_t stream);

}  // namespace lyradiff