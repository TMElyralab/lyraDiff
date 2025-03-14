#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"

namespace lyradiff {

template<typename T>
__global__ void loadLora(T* dst, const T* src, const size_t size, float alpha)
{
    T tmp_lora = cuda_cast<T>(alpha);
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] += src[tid] * tmp_lora;
        // dst[tid] += src[tid];
    }
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
                         __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

template<typename T>
__global__ void getNewWeightScale(float*               new_weight_scale,
                                  const __nv_fp8_e4m3* weight,
                                  const float*         prev_weight_scale,
                                  const T*             lora_weight,
                                  const size_t         size,
                                  float                alpha)
{
    float prev_scale = prev_weight_scale[0];
    T     tmp_alpha  = cuda_cast<T>(alpha);
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        // T tmp = cuda_cast<T>(cuda_cast<float>(weight[tid]) * prev_scale) + lora_weight[tid] * tmp_alpha;

        // atomicMaxFloat(new_weight_scale, cuda_abs<float>(cuda_cast<float>(tmp)) / 448.0);

        float tmp_f = cuda_cast<float>(weight[tid]) * prev_scale + cuda_cast<float>(lora_weight[tid]) * alpha;

        atomicMaxFloat(new_weight_scale, cuda_abs<float>(tmp_f / 448.0));

        // dst[tid] += src[tid];
    }
}

template<typename T>
__global__ void getFP8WeightScale(float* new_weight_scale, const T* weight, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        atomicMaxFloat(new_weight_scale, cuda_abs<float>(cuda_cast<float>(weight[tid]) / 448.0));
    }
}

template<typename T>
__global__ void loadLoraFP8(__nv_fp8_e4m3* weight,
                            const float*   prev_weight_scale,
                            const float*   new_weight_scale,
                            const T*       lora_weight,
                            const size_t   size,
                            float          alpha)
{
    // T tmp_lora = cuda_cast<T>(alpha);
    T     tmp_alpha  = cuda_cast<T>(alpha);
    float new_scale  = new_weight_scale[0];
    float prev_scale = prev_weight_scale[0];
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        // float tmp =
        //     cuda_cast<float>(cuda_cast<T>(cuda_cast<float>(weight[tid]) * prev_scale) + lora_weight[tid] *
        //     tmp_alpha);
        float tmp_f = cuda_cast<float>(weight[tid]) * prev_scale + cuda_cast<float>(lora_weight[tid]) * alpha;

        weight[tid] = cuda_cast<__nv_fp8_e4m3>(tmp_f / new_scale);
    }
}

template<typename T>
void invokeLoadLora(T* dst, const T* src, const size_t size, float alpha, cudaStream_t stream)
{
    loadLora<<<256, 256, 0, stream>>>(dst, src, size, alpha);
}

template<typename T>
void calculateNewWeightScale(float*               new_weight_scale,
                             const __nv_fp8_e4m3* weight,
                             const float*         prev_weight_scale,
                             const T*             lora_weight,
                             const size_t         size,
                             float                alpha,
                             cudaStream_t         stream)
{
    getNewWeightScale<<<256, 256, 0, stream>>>(new_weight_scale, weight, prev_weight_scale, lora_weight, size, alpha);
}

template<typename T>
void invokeLoadLoraFP8(__nv_fp8_e4m3* weight,
                       const float*   prev_weight_scale,
                       const float*   new_weight_scale,
                       const T*       lora_weight,
                       const size_t   size,
                       float          alpha,
                       cudaStream_t   stream)
{
    loadLoraFP8<<<256, 256, 0, stream>>>(weight, prev_weight_scale, new_weight_scale, lora_weight, size, alpha);
}

template<typename T>
void invokeGetFP8WeightScale(float* new_weight_scale, const T* weight, const size_t size, cudaStream_t stream)
{
    getFP8WeightScale<<<256, 256, 0, stream>>>(new_weight_scale, weight, size);
}

#define INSTANTIATE_INVOKE_LOAD_LORA(T)                                                                                \
    template void invokeLoadLora(T* dst, const T* src, const size_t size, float alpha, cudaStream_t stream)

INSTANTIATE_INVOKE_LOAD_LORA(float);
INSTANTIATE_INVOKE_LOAD_LORA(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_LOAD_LORA(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_LOAD_LORA

#define INSTANTIATE_INVOKE_LOAD_LORA_FP8(T)                                                                            \
    template void invokeLoadLoraFP8(__nv_fp8_e4m3* weight,                                                             \
                                    const float*   prev_weight_scale,                                                  \
                                    const float*   new_weight_scale,                                                   \
                                    const T*       lora_weight,                                                        \
                                    const size_t   size,                                                               \
                                    float          alpha,                                                              \
                                    cudaStream_t   stream)

INSTANTIATE_INVOKE_LOAD_LORA_FP8(float);
INSTANTIATE_INVOKE_LOAD_LORA_FP8(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_LOAD_LORA_FP8(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_LOAD_LORA_FP8

#define INSTANTIATE_CALCULATE_NEW_WEIGHT_SCALE(T)                                                                      \
    template void calculateNewWeightScale(float*               new_weight_scale,                                       \
                                          const __nv_fp8_e4m3* weight,                                                 \
                                          const float*         prev_weight_scale,                                      \
                                          const T*             lora_weight,                                            \
                                          const size_t         size,                                                   \
                                          float                alpha,                                                  \
                                          cudaStream_t         stream)
INSTANTIATE_CALCULATE_NEW_WEIGHT_SCALE(float);
INSTANTIATE_CALCULATE_NEW_WEIGHT_SCALE(half);

#ifdef ENABLE_BF16
INSTANTIATE_CALCULATE_NEW_WEIGHT_SCALE(__nv_bfloat16);
#endif
#undef INSTANTIATE_CALCULATE_NEW_WEIGHT_SCALE

#define INSTANTIATE_INVOKE_GET_FP8_WEIGHT_SCALE(T)                                                                     \
    template void invokeGetFP8WeightScale(                                                                             \
        float* new_weight_scale, const T* weight, const size_t size, cudaStream_t stream)

INSTANTIATE_INVOKE_GET_FP8_WEIGHT_SCALE(float);
INSTANTIATE_INVOKE_GET_FP8_WEIGHT_SCALE(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_GET_FP8_WEIGHT_SCALE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_GET_FP8_WEIGHT_SCALE
}  // namespace lyradiff