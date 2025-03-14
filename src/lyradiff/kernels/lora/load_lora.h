#pragma once

#include "src/lyradiff/utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeLoadLora(T* dst, const T* src, const size_t size, float alpha, cudaStream_t stream = 0);

template<typename T>
void calculateNewWeightScale(float*               new_weight_scale,
                             const __nv_fp8_e4m3* weight,
                             const float*         prev_weight_scale,
                             const T*             lora_weight,
                             const size_t         size,
                             float                alpha,
                             cudaStream_t         stream = 0);

template<typename T>
void invokeLoadLoraFP8(__nv_fp8_e4m3* weight,
                       const float*   prev_weight_scale,
                       const float*   new_weight_scale,
                       const T*       lora_weight,
                       const size_t   size,
                       float          alpha,
                       cudaStream_t   stream = 0);

template<typename T>
void invokeGetFP8WeightScale(float* new_weight_scale, const T* weight, const size_t size, cudaStream_t stream = 0);

}  // namespace lyradiff
