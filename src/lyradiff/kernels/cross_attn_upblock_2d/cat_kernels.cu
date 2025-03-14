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
template<typename T>
__global__ void CatByChannel(T* dst, const T* src1, const T* src2, const size_t channel1, const size_t channel2)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 4;
    int all_dim = channel1 + channel2;

    using T4 = typename TypeConverter4<T>::Type;  // float to float2, half to half2

    if (cur_col < channel1) {
        T4 ele2 = *reinterpret_cast<const T4*>(src1 + cur_row * channel1 + cur_col);
        *reinterpret_cast<T4*>(&dst[cur_row * all_dim + cur_col]) = ele2;
    }
    else {
        T4 ele2 = *reinterpret_cast<const T4*>(src2 + cur_row * channel2 + cur_col - channel1);
        *reinterpret_cast<T4*>(&dst[cur_row * all_dim + cur_col]) = ele2;
    }
}

template<typename T>
void invokeCatByChannel(T*           dst,
                        const T*     src1,
                        const T*     src2,
                        const size_t channel1,
                        const size_t channel2,
                        const size_t height,
                        const size_t width,
                        const int    batch_size,
                        cudaStream_t stream)
{
    int grid             = batch_size * height * width;
    int thread_per_block = (channel1 + channel2) / 4;
    CatByChannel<T><<<grid, thread_per_block, 0, stream>>>(dst, src1, src2, channel1, channel2);
}

#define INSTANTIATE_INVOKE_CAT_BY_CHANNEL(T)                                                                           \
    template void invokeCatByChannel(T*           dst,                                                                 \
                                     const T*     src1,                                                                \
                                     const T*     src2,                                                                \
                                     const size_t channel1,                                                            \
                                     const size_t channel2,                                                            \
                                     const size_t height,                                                              \
                                     const size_t width,                                                               \
                                     const int    batch_size,                                                          \
                                     cudaStream_t stream)

INSTANTIATE_INVOKE_CAT_BY_CHANNEL(float);
INSTANTIATE_INVOKE_CAT_BY_CHANNEL(half);
#undef INSTANTIATE_INVOKE_CAT_BY_CHANNEL

}  // namespace lyradiff