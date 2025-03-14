#include "tensor_helper.h"

namespace lyradiff {
template<typename T_OUT, typename T_IN>
__global__ void tensorD2DConvert(T_OUT* dst, const T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        if (std::is_same<T_IN, float>::value && std::is_same<T_OUT, half>::value) {
            dst[tid] = __float2half(src[tid]);
        }
        if (std::is_same<T_IN, half>::value && std::is_same<T_OUT, float>::value) {
            dst[tid] = __half2float(src[tid]);
        }
    }
}

template<typename T_OUT, typename T_IN>
void invokeTensorD2DConvert(T_OUT* tgt, const T_IN* src, const size_t size, cudaStream_t stream)
{
    tensorD2DConvert<<<256, 256, 0, stream>>>(tgt, src, size);
}

template void invokeTensorD2DConvert<half, float>(half* tgt, const float* src, const size_t size, cudaStream_t stream);
template void invokeTensorD2DConvert<float, half>(float* tgt, const half* src, const size_t size, cudaStream_t stream);
}  // namespace lyradiff
