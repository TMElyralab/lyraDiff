#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"

namespace lyradiff {

template<typename T>
__global__ void mulGammaAndAddBeta(T* dst, const T* src, const T* gamma, const T* beta, const size_t channels)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 4;

    using T4      = typename TypeConverter4<T>::Type;  // float to float4, half to half4
    T4 src_ele4   = *reinterpret_cast<const T4*>(src + cur_row * channels + cur_col);
    T4 gamma_ele4 = *reinterpret_cast<const T4*>(gamma + cur_row * channels + cur_col);
    T4 beta_ele4  = *reinterpret_cast<const T4*>(beta + cur_row * channels + cur_col);

    T4 ele4;
    T tmp = cuda_cast<T>(1.0);
    ele4.x = src_ele4.x * (gamma_ele4.x + tmp) + beta_ele4.x;
    ele4.y = src_ele4.y * (gamma_ele4.y + tmp) + beta_ele4.y;
    ele4.z = src_ele4.z * (gamma_ele4.z + tmp) + beta_ele4.z;
    ele4.w = src_ele4.w * (gamma_ele4.w + tmp) + beta_ele4.w;

    *reinterpret_cast<T4*>(&dst[cur_row * channels + cur_col]) = ele4;
}

template<typename T>
__global__ void mulControlScale(T* dst, const T* src, const T* src2, const size_t channels, const float control_scale)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 4;

    using T4     = typename TypeConverter4<T>::Type;  // float to float2, half to half2
    T4 src1_ele4 = *reinterpret_cast<const T4*>(src + cur_row * channels + cur_col);
    T4 src2_ele4 = *reinterpret_cast<const T4*>(src2 + cur_row * channels + cur_col);

    T4 ele4;
    T tmp = cuda_cast<T>(control_scale);
    T tmp2 = cuda_cast<T>(1 - control_scale);
    ele4.x = src1_ele4.x * tmp + src2_ele4.x * tmp2;
    ele4.y = src1_ele4.y * tmp + src2_ele4.y * tmp2;
    ele4.z = src1_ele4.z * tmp + src2_ele4.z * tmp2;
    ele4.w = src1_ele4.w * tmp + src2_ele4.w * tmp2;

    *reinterpret_cast<T4*>(&dst[cur_row * channels + cur_col]) = ele4;
}

template<typename T>
void invokeMulGammaAndAddBeta(T*           dst,
                              const T*     src,
                              const T*     gamma,
                              const T*     beta,
                              const size_t batch_size,
                              const size_t height,
                              const size_t width,
                              const size_t channels,
                              cudaStream_t stream = 0)
{
    int grid             = batch_size * height * width;
    int thread_per_block = channels / 4;
    mulGammaAndAddBeta<T><<<grid, thread_per_block, 0, stream>>>(dst, src, gamma, beta, channels);
}

template<typename T>
void invokeMulControlScale(T*           dst,
                           const T*     src,
                           const T*     src2,
                           const size_t batch_size,
                           const size_t height,
                           const size_t width,
                           const size_t channels,
                           const float  control_scale,
                           cudaStream_t stream = 0)
{
    int grid             = batch_size * height * width;
    int thread_per_block = (channels) / 4;
    mulControlScale<T><<<grid, thread_per_block, 0, stream>>>(dst, src, src2, channels, control_scale);
}

#define INSTANTIATE_INVOKE_MUL_CONTROL_SCALE(T)                                                                        \
    template void invokeMulControlScale(T*           dst,                                                              \
                                        const T*     src,                                                              \
                                        const T*     src2,                                                             \
                                        const size_t batch_size,                                                       \
                                        const size_t height,                                                           \
                                        const size_t width,                                                            \
                                        const size_t channels,                                                         \
                                        const float  control_scale,                                                    \
                                        cudaStream_t stream = 0)

INSTANTIATE_INVOKE_MUL_CONTROL_SCALE(float);
INSTANTIATE_INVOKE_MUL_CONTROL_SCALE(half);
#undef INSTANTIATE_INVOKE_MUL_CONTROL_SCALE

#define INSTANTIATE_INVOKE_MUL_GAMMA_AND_BETA(T)                                                                        \
    template void invokeMulGammaAndAddBeta(T*           dst,                                                           \
                                           const T*     src,                                                           \
                                           const T*     gamma,                                                         \
                                           const T*     beta,                                                          \
                                           const size_t batch_size,                                                    \
                                           const size_t height,                                                        \
                                           const size_t width,                                                         \
                                           const size_t channels,                                                      \
                                           cudaStream_t stream = 0)

INSTANTIATE_INVOKE_MUL_GAMMA_AND_BETA(float);
INSTANTIATE_INVOKE_MUL_GAMMA_AND_BETA(half);
#undef INSTANTIATE_INVOKE_MUL_GAMMA_AND_BETA
}  // namespace lyradiff