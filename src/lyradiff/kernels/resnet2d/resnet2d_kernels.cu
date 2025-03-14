#include "resnet2d_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include <cassert>

namespace lyradiff {

template<typename T>
__global__ void addConvAndTemb(T* dst, const T* conv_out, const T* temb_out, const T* temb_bias, const size_t nchannels)
{
    // conv_out: [B, H, W, C] like: [2, 16, 16, 1280]
    // temb_out: [B, H, W, C] like: [2, 1, 1, 1280] ---> ptr: [2, 1280]
    // temb_bias: [C] like: [1280]

    // 这个 kernel 使用限定条件：nchannels 维度小于 2048，最好 nchannels 能被 2 整除
    // Grid: [B, H, W]
    // Block: [C]
    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    int batch_id         = blockIdx.x;
    int block_id         = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int conv_out_offset  = block_id * nchannels + threadIdx.x * 2;
    int temb_out_offset  = batch_id * nchannels + threadIdx.x * 2;
    int temb_bias_offset = threadIdx.x * 2;

    if (temb_bias_offset < nchannels) {
        T2 conv_out_2  = *reinterpret_cast<const T2*>(conv_out + conv_out_offset);
        T2 temb_out_2  = *reinterpret_cast<const T2*>(temb_out + temb_out_offset);
        T2 temb_bias_2 = *reinterpret_cast<const T2*>(temb_bias + temb_bias_offset);

        T2 res2;
        res2.x = conv_out_2.x + temb_out_2.x + temb_bias_2.x;
        res2.y = conv_out_2.y + temb_out_2.y + temb_bias_2.y;

        *reinterpret_cast<T2*>(&dst[conv_out_offset]) = res2;
    }
}

template<typename T>
void invokeAddConvAndTemb(T*           dst,
                          const T*     conv_out,
                          const T*     temb_out,
                          const T*     temb_bias,
                          const size_t batch_size,
                          const size_t height,
                          const size_t width,
                          const size_t nchannels,
                          cudaStream_t stream)
{
    assert(nchannels <= 2048);
    dim3 grid(batch_size, height, width);
    dim3 block((nchannels + 2 - 1) / 2);
    // printf("%ld %ld %ld %ld\n", batch_size, height, width, nchannels);
    // printf("grid: %ld %ld %ld\n", grid.x, grid.y, grid.z);
    // printf("block: %ld %ld %ld\n", block.x, block.y, block.z);
    addConvAndTemb<T><<<grid, block, 0, stream>>>(dst, conv_out, temb_out, temb_bias, nchannels);
}

// 为 float 和 half 做模板特化
#define INSTANTIATE_INVOKE_ADD_CONV_AND_TEMB(T)                                                                        \
    template void invokeAddConvAndTemb(T*           dst,                                                               \
                                       const T*     conv_out,                                                          \
                                       const T*     temb_out,                                                          \
                                       const T*     temb_bias,                                                         \
                                       const size_t batch_size,                                                        \
                                       const size_t height,                                                            \
                                       const size_t width,                                                             \
                                       const size_t nchannels,                                                         \
                                       cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_CONV_AND_TEMB(float);
INSTANTIATE_INVOKE_ADD_CONV_AND_TEMB(half);
#undef INSTANTIATE_INVOKE_ADD_CONV_AND_TEMB

template<typename T>
__global__ void
addTwoConvOutScale(T* dst, const T* conv1_out, const T* conv2_out, const size_t nchannels, const float scale)
{
    // conv1_out: [B, H, W, C] like: [2, 16, 16, 1280]
    // conv2_out: [B, H, W, C] like: [2, 16, 16, 1280]

    // 这个 kernel 使用限定条件：nchannels 维度小于 2048，最好 nchannels 能被 2 整除
    // Grid: [B, H, W]
    // Block: [C]
    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    int block_id        = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int conv_out_offset = block_id * nchannels + threadIdx.x * 2;

    if (threadIdx.x * 2 < nchannels) {
        T2 conv1_out_2 = *reinterpret_cast<const T2*>(conv1_out + conv_out_offset);
        T2 conv2_out_2 = *reinterpret_cast<const T2*>(conv2_out + conv_out_offset);

        T2 res2;
        res2.x = (conv1_out_2.x + conv2_out_2.x) / static_cast<T>(scale);
        res2.y = (conv1_out_2.y + conv2_out_2.y) / static_cast<T>(scale);

        *reinterpret_cast<T2*>(&dst[conv_out_offset]) = res2;
    }
}

template<typename T>
void invokeAddTwoConvOutScale(T*           dst,
                              const T*     conv1_out,
                              const T*     conv2_out,
                              const size_t batch_size,
                              const size_t height,
                              const size_t width,
                              const size_t nchannels,
                              const float  scale,
                              cudaStream_t stream)
{
    assert(nchannels <= 2048);
    dim3 grid(batch_size, height, width);
    dim3 block((nchannels + 2 - 1) / 2);

    // printf("%ld %ld %ld %ld\n", batch_size, height, width, nchannels);
    // printf("grid: %ld %ld %ld\n", grid.x, grid.y, grid.z);
    // printf("block: %ld %ld %ld\n", block.x, block.y, block.z);

    addTwoConvOutScale<T><<<grid, block, 0, stream>>>(dst, conv1_out, conv2_out, nchannels, scale);
}

// 为 float 和 half 做模板特化
#define INSTANTIATE_INVOKE_ADD_TWO_CONV_OUT_SCALE(T)                                                                   \
    template void invokeAddTwoConvOutScale(T*           dst,                                                           \
                                           const T*     conv1_out,                                                     \
                                           const T*     conv2_out,                                                     \
                                           const size_t batch_size,                                                    \
                                           const size_t height,                                                        \
                                           const size_t width,                                                         \
                                           const size_t nchannels,                                                     \
                                           const float  scale,                                                         \
                                           cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_TWO_CONV_OUT_SCALE(float);
INSTANTIATE_INVOKE_ADD_TWO_CONV_OUT_SCALE(half);
#undef INSTANTIATE_INVOKE_ADD_TWO_CONV_OUT_SCALE

}  // namespace lyradiff
