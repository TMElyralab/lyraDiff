#pragma once

#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

/* Common util */
__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

/* Gelu Activation */
template<typename T>
struct GeluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T val)
    {
        const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
        return val * cdf;
    }
};

template<>
struct GeluActivation<half> {
    using return_type = half;
    static __device__ __forceinline__ half apply(const half val)
    {
        float tmp_val  = __half2float(val);
        half  tmp_pow  = val * val * val;
        float tmp_pow2 = __half2float(tmp_pow);

        float tmp = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp_val + 0.044715f * tmp_pow2))));
        return val * __float2half(tmp);
    }
};

template<>
struct GeluActivation<float2> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const float2 val)
    {
        float2 tmp_pow = val * val * val;
        float2 tmp(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return val * tmp;
    }
};

template<>
struct GeluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val)
    {
        half2  val_pow3 = __hmul2(val, __hmul2(val, val));
        float2 tmp_pow  = __half22float2(val_pow3);
        float2 tmp      = __half22float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return __hmul2(val, __float22half2_rn(tmp));
    }
};

#ifdef ENABLE_BF16
template<>
struct GeluActivation<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val)
    {
        __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
        float2         tmp_pow  = bf1622float2(val_pow3);
        float2         tmp      = bf1622float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
    }
};
#endif

/* Silu Activation */
template<typename T>
struct SiluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        return (T)((float)val / (1.0f + __expf((float)-val)));
    }
};

template<>
struct SiluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val)
    {
        float2 tmp_val = __half22float2(val);
        tmp_val.x      = SiluActivation<float>::apply(tmp_val.x);
        tmp_val.y      = SiluActivation<float>::apply(tmp_val.y);

        return cuda_cast<half2, float2>(tmp_val);
    }
};

template<>
struct SiluActivation<float2> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const float2& val)
    {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};

#ifdef ENABLE_BF16
template<>
struct SiluActivation<__nv_bfloat162> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const __nv_bfloat162& val)
    {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};

template<>
struct SiluActivation<__nv_bfloat16> {
    using return_type = __nv_bfloat16;
    static __device__ __forceinline__ __nv_bfloat16 apply(const __nv_bfloat16& val)
    {
        return __float2bfloat16(SiluActivation<float>::apply(__bfloat162float(val)));
    }
};
#endif  // ENABLE_BF16

/* Identity Activation */
template<typename T>
struct IdentityActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val;
    }
};

template<template<typename T> class Activation, typename T>
void invokeGenericActivation(T* dst, const T* src, const size_t length, cudaStream_t stream);

}  // namespace lyradiff
