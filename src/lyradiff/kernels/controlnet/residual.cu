#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

using namespace std;

namespace lyradiff {
// input format NHWC
__global__ void AddResidual(half* dst, const half* src1, const half* src2, const float scale, const size_t channel)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 2;

    // using T4 = typename TypeConverter4<T>::Type; // float to float2, half to half2

    half2 ele1 = *reinterpret_cast<const half2*>(src1 + cur_row * channel + cur_col);
    half2 ele2 = *reinterpret_cast<const half2*>(src2 + cur_row * channel + cur_col);

    float2 tmp1 = __half22float2(ele1);
    float2 tmp2 = __half22float2(ele2);

    tmp1.x += tmp2.x * scale;
    tmp1.y += tmp2.y * scale;

    *reinterpret_cast<half2*>(&dst[cur_row * channel + cur_col]) = __float22half2_rn(tmp1);
}

__global__ void AddResidual(float* dst, const float* src1, const float* src2, const float scale, const size_t channel)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 2;

    // using T4 = typename TypeConverter4<T>::Type; // float to float2, half to half2

    float2 ele1 = *reinterpret_cast<const float2*>(src1 + cur_row * channel + cur_col);
    float2 ele2 = *reinterpret_cast<const float2*>(src2 + cur_row * channel + cur_col);

    ele1.x += ele2.x * scale;
    ele1.y += ele2.y * scale;

    *reinterpret_cast<float2*>(&dst[cur_row * channel + cur_col]) = ele1;
}

// input format NHWC
__global__ void AddResidualWithDifferentBatch(
    half* dst, const half* src1, const half* src2, const float scale, const size_t channel, const size_t src2_size)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 2;

    // using T4 = typename TypeConverter4<T>::Type; // float to float2, half to half2

    half2 ele1 = *reinterpret_cast<const half2*>(src1 + cur_row * channel + cur_col);
    half2 ele2 = *reinterpret_cast<const half2*>(src2 + (cur_row * channel + cur_col) % src2_size);

    float2 tmp1 = __half22float2(ele1);
    float2 tmp2 = __half22float2(ele2);

    tmp1.x += tmp2.x * scale;
    tmp1.y += tmp2.y * scale;

    *reinterpret_cast<half2*>(&dst[cur_row * channel + cur_col]) = __float22half2_rn(tmp1);
}

__global__ void AddResidualWithDifferentBatch(
    float* dst, const float* src1, const float* src2, const float scale, const size_t channel, const size_t src2_size)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 2;

    // using T4 = typename TypeConverter4<T>::Type; // float to float2, half to half2

    float2 ele1 = *reinterpret_cast<const float2*>(src1 + cur_row * channel + cur_col);
    float2 ele2 = *reinterpret_cast<const float2*>(src2 + (cur_row * channel + cur_col) % src2_size);

    ele1.x += ele2.x * scale;
    ele1.y += ele2.y * scale;

    *reinterpret_cast<float2*>(&dst[cur_row * channel + cur_col]) = ele1;
}

template<typename T>
void invokeAddResidual(T*           dst,
                       const T*     src1,
                       const T*     src2,
                       const float  scale,
                       const size_t channel,
                       const size_t height,
                       const size_t width,
                       const int    batch_size,
                       cudaStream_t stream)
{
    int grid             = batch_size * height * width;
    int thread_per_block = channel / 2;
    AddResidual<<<grid, thread_per_block, 0, stream>>>(dst, src1, src2, scale, channel);
}


template<typename T>
void invokeAddResidualWithDifferentBatch(T*           dst,
                                         const T*     src1,
                                         const T*     src2,
                                         const float  scale,
                                         const size_t channel,
                                         const size_t height,
                                         const size_t width,
                                         const int    batch_size,
                                         const int    batch_size_2,
                                         cudaStream_t stream)
{
    int    grid             = batch_size * height * width;
    size_t src2_size        = batch_size_2 * height * width * channel;
    int    thread_per_block = channel / 2;
    AddResidualWithDifferentBatch<<<grid, thread_per_block, 0, stream>>>(dst, src1, src2, scale, channel, src2_size);
}

#define INSTANTIATE_INVOKE_ADD_RESIDUAL(T)                                                                             \
    template void invokeAddResidual(T*           dst,                                                                  \
                                    const T*     src1,                                                                 \
                                    const T*     src2,                                                                 \
                                    const float  scale,                                                                \
                                    const size_t channel,                                                              \
                                    const size_t height,                                                               \
                                    const size_t width,                                                                \
                                    const int    batch_size,                                                           \
                                    cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_RESIDUAL(float);
INSTANTIATE_INVOKE_ADD_RESIDUAL(half);
#undef INSTANTIATE_INVOKE_ADD_RESIDUAL

#define INSTANTIATE_INVOKE_ADD_RESIDUAL_WITH_DIFFERENT_BATCH(T)                                                        \
    template void invokeAddResidualWithDifferentBatch(T*           dst,                                                \
                                                      const T*     src1,                                               \
                                                      const T*     src2,                                               \
                                                      const float  scale,                                              \
                                                      const size_t channel,                                            \
                                                      const size_t height,                                             \
                                                      const size_t width,                                              \
                                                      const int    batch_size,                                         \
                                                      const int    batch_size_2,                                       \
                                                      cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_RESIDUAL_WITH_DIFFERENT_BATCH(float);
INSTANTIATE_INVOKE_ADD_RESIDUAL_WITH_DIFFERENT_BATCH(half);
#undef INSTANTIATE_INVOKE_ADD_RESIDUAL_WITH_DIFFERENT_BATCH

}  // namespace lyradiff