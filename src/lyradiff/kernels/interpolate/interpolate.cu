#include "src/lyradiff/kernels/interpolate/interpolate.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"
#include <cuda_fp16.h>

namespace lyradiff {

template<typename T, size_t N>
__global__ void interpolateNearestKernel(T* dst, const T* src, size_t bs, size_t height, size_t width, size_t channels)
{
    using T2 = typename TypeConverter<T>::Type;

    int64_t pixel_idx  = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    size_t  height_pos = blockIdx.y;
    size_t  width_pos  = blockIdx.z;

    int64_t src_idx = blockIdx.x * height * width * channels + (height_pos / 2) * width * channels
                      + (width_pos / 2) * channels + N * threadIdx.x;
    int64_t dst_idx = pixel_idx * channels + N * threadIdx.x;
#pragma unroll
    for (int i = 0; i < N; i += 2) {
        if (N * threadIdx.x + i < channels) {
            T2 ele2                                   = *reinterpret_cast<const T2*>(src + i + src_idx);
            *reinterpret_cast<T2*>(&dst[dst_idx + i]) = ele2;
        }
    }
}

template<typename T>
void invokeInterpolateNearestToShape(T*           dst,
                                     const T*     src,
                                     size_t       bs,
                                     size_t       height,
                                     size_t       width,
                                     size_t       channels,
                                     size_t       tgt_height,
                                     size_t       tgt_width,
                                     cudaStream_t stream)
{

    dim3 grid(bs, tgt_height, tgt_width);
    dim3 block(ceil((channels + 1) / 2) );

    interpolateNearestKernel<T, 2><<<grid, block, 0, stream>>>(dst, src, bs, height, width, channels);
}

#define INSTANTIATE_INVOKEINTERPOLATENEARESTTOSHAPE(T)                                                                 \
    template void invokeInterpolateNearestToShape(T*           dst,                                                    \
                                                  const T*     src,                                                    \
                                                  size_t       bs,                                                     \
                                                  size_t       height,                                                 \
                                                  size_t       width,                                                  \
                                                  size_t       channels,                                               \
                                                  size_t       tgt_height,                                             \
                                                  size_t       tgt_width,                                              \
                                                  cudaStream_t stream)

INSTANTIATE_INVOKEINTERPOLATENEARESTTOSHAPE(float);
INSTANTIATE_INVOKEINTERPOLATENEARESTTOSHAPE(half);
#undef INSTANTIATE_INVOKEINTERPOLATENEARESTTOSHAPE

template<typename T>
void invokeInterpolateNearest(T*           dst,
                              const T*     src,
                              size_t       bs,
                              size_t       height,
                              size_t       width,
                              size_t       channels,
                              size_t       up_factor,
                              cudaStream_t stream)
{

    dim3 grid(bs, height * up_factor, width * up_factor);
    dim3 block(ceil((channels + 1) / 2) );

    interpolateNearestKernel<T, 2><<<grid, block, 0, stream>>>(dst, src, bs, height, width, channels);
}

#define INSTANTIATE_INVOKEINTERPOLATENEAREST(T)                                                                        \
    template void invokeInterpolateNearest(T*           dst,                                                           \
                                           const T*     src,                                                           \
                                           size_t       bs,                                                            \
                                           size_t       height,                                                        \
                                           size_t       width,                                                         \
                                           size_t       channels,                                                      \
                                           size_t       up_factor,                                                     \
                                           cudaStream_t stream)

INSTANTIATE_INVOKEINTERPOLATENEAREST(float);
INSTANTIATE_INVOKEINTERPOLATENEAREST(half);
#undef INSTANTIATE_INVOKEINTERPOLATENEAREST

}  // namespace lyradiff
