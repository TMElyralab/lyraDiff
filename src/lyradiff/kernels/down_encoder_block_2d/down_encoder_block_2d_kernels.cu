#include "down_encoder_block_2d_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include <cassert>

namespace lyradiff {

template<typename T>
__global__ void downSamplePad(
    T* dst, const T* input, const size_t batch_size, const size_t height, const size_t width, const size_t nchannels)
{
    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    int batch_id     = blockIdx.x;
    int ori_block_id = batch_id * height * width + blockIdx.y * width + blockIdx.z;
    int dst_block_id = batch_id * (height + 1) * (width + 1) + blockIdx.y * (width + 1) + blockIdx.z;

    int ori_offset = ori_block_id * nchannels + threadIdx.x * 2;
    int dst_offset = dst_block_id * nchannels + threadIdx.x * 2;

    T2 ori_2 = *reinterpret_cast<const T2*>(input + ori_offset);

    *reinterpret_cast<T2*>(&dst[dst_offset]) = ori_2;
}

template<typename T>
void invokeDownSamplePad(T*           dst,
                         const T*     input,
                         const size_t batch_size,
                         const size_t height,
                         const size_t width,
                         const size_t nchannels,
                         cudaStream_t stream)
{
    dim3 grid(batch_size, height, width);
    dim3 block(nchannels / 2);
    // printf("%ld %ld %ld %ld\n", batch_size, height, width, nchannels);
    // printf("grid: %ld %ld %ld\n", grid.x, grid.y, grid.z);
    // printf("block: %ld %ld %ld\n", block.x, block.y, block.z);
    downSamplePad<T><<<grid, block, 0, stream>>>(dst, input, batch_size, height, width, nchannels);
}

// 为 float 和 half 做模板特化
#define INSTANTIATE_INVOKE_DOWN_SAMPLE_PAD(T)                                                                          \
    template void invokeDownSamplePad(T*           dst,                                                                \
                                      const T*     input,                                                              \
                                      const size_t batch_size,                                                         \
                                      const size_t height,                                                             \
                                      const size_t width,                                                              \
                                      const size_t nchannels,                                                          \
                                      cudaStream_t stream)

INSTANTIATE_INVOKE_DOWN_SAMPLE_PAD(float);
INSTANTIATE_INVOKE_DOWN_SAMPLE_PAD(half);
#undef INSTANTIATE_INVOKE_DOWN_SAMPLE_PAD

}  // namespace lyradiff
