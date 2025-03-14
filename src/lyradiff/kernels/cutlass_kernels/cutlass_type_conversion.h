

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tllm to Cutlass

template<typename T>
struct TllmToCutlassTypeAdapter {
    using type = T;
};

template<>
struct TllmToCutlassTypeAdapter<half> {
    using type = cutlass::half_t;
};

#if defined(ENABLE_BF16)
template<>
struct TllmToCutlassTypeAdapter<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};
#endif

#if defined(ENABLE_FP8)
template<>
struct TllmToCutlassTypeAdapter<__nv_fp8_e4m3> {
    using type = cutlass::float_e4m3_t;
};

template<>
struct TllmToCutlassTypeAdapter<__nv_fp8_e5m2> {
    using type = cutlass::float_e5m2_t;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cutlass to Tllm

template<typename T>
struct CutlassToTllmTypeAdapter {
    using type = T;
};

template<>
struct CutlassToTllmTypeAdapter<cutlass::half_t> {
    using type = half;
};

#if defined(ENABLE_BF16)
template<>
struct CutlassToTllmTypeAdapter<cutlass::bfloat16_t> {
    using type = __nv_bfloat16;
};
#endif

#if defined(ENABLE_FP8)
template<>
struct CutlassToTllmTypeAdapter<cutlass::float_e4m3_t> {
    using type = __nv_fp8_e4m3;
};

template<>
struct CutlassToTllmTypeAdapter<cutlass::float_e5m2_t> {
    using type = __nv_fp8_e5m2;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
