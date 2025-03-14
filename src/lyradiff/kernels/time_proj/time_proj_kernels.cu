#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"
#include "time_proj_kernels.h"

using namespace std;

namespace lyradiff {
// Time Projection

// template<typename T>
// __global__ void TimeProjection(T* dst, const float& timestep, const float* exponents, const size_t sin_offset, const
// size_t cos_offset, const size_t half_dim_){
//     // grid x 维度代表, batch_idx
//     // block x 维度为 half_dim_ / 2
//     int dst_offset = blockIdx.x * half_dim_ * 2 + threadIdx.x * 2;

//     size_t channel_idx_0 = threadIdx.x * 2;
//     size_t channel_idx_1 = threadIdx.x * 2 + 1;
//     float2 exponent_factor_2 = make_float2( exponents[channel_idx_0] * (timestep), exponents[channel_idx_1] *
//     (timestep) );

//     float2 sin_temb;
//     sin_temb.x = sinf( exponent_factor_2.x );
//     sin_temb.y = sinf( exponent_factor_2.y );

//     float2 cos_temb;
//     cos_temb.x = cosf( exponent_factor_2.x );
//     cos_temb.y = cosf( exponent_factor_2.y );

//     using T2 = typename TypeConverter<T>::Type; // float to float2, half to half2

//     if (std::is_same<T, half>::value){
//         *reinterpret_cast<T2*>(&dst[dst_offset + sin_offset]) = cuda_cast<T2, float2>(sin_temb);
//         *reinterpret_cast<T2*>(&dst[dst_offset + cos_offset]) = cuda_cast<T2, float2>(cos_temb);
//     }
//     else{
//         *reinterpret_cast<T2*>(&dst[dst_offset + sin_offset]) = sin_temb;
//         *reinterpret_cast<T2*>(&dst[dst_offset + cos_offset]) = cos_temb;
//     }
// }

// __global__ void TimeProjection(float*       dst,
//                                const float* timestep,
//                                const float* exponents,
//                                const size_t sin_offset,
//                                const size_t cos_offset,
//                                const size_t half_dim_)
// {
//     // grid x 维度代表, batch_idx
//     // block x 维度为 half_dim_ / 2
//     int dst_offset = blockIdx.x * half_dim_ * 2 + threadIdx.x * 2;

//     size_t channel_idx_0 = threadIdx.x * 2;
//     size_t channel_idx_1 = threadIdx.x * 2 + 1;
//     float2 exponent_factor_2 =
//         make_float2(exponents[channel_idx_0] * (*timestep), exponents[channel_idx_1] * (*timestep));

//     float2 sin_temb;
//     sin_temb.x = sinf(exponent_factor_2.x);
//     sin_temb.y = sinf(exponent_factor_2.y);

//     float2 cos_temb;
//     cos_temb.x = cosf(exponent_factor_2.x);
//     cos_temb.y = cosf(exponent_factor_2.y);

//     *reinterpret_cast<float2*>(&dst[dst_offset + sin_offset]) = sin_temb;
//     *reinterpret_cast<float2*>(&dst[dst_offset + cos_offset]) = cos_temb;
// }

template<typename T>
__global__ void TimeProjection(T*           dst,
                               const float* timestep,
                               const float* exponents,
                               const size_t sin_offset,
                               const size_t cos_offset,
                               const size_t half_dim_)
{
    // grid x 维度代表, batch_idx
    // block x 维度为 half_dim_ / 2
    using T2 = typename TypeConverter<T>::Type;

    int    dst_offset    = blockIdx.x * half_dim_ * 2 + threadIdx.x * 2;
    size_t channel_idx_0 = threadIdx.x * 2;
    size_t channel_idx_1 = threadIdx.x * 2 + 1;
    float2 exponent_factor_2 =
        make_float2(exponents[channel_idx_0] * (*timestep), exponents[channel_idx_1] * (*timestep));

    float2 sin_temb;
    sin_temb.x = sinf(exponent_factor_2.x);
    sin_temb.y = sinf(exponent_factor_2.y);

    float2 cos_temb;
    cos_temb.x = cosf(exponent_factor_2.x);
    cos_temb.y = cosf(exponent_factor_2.y);

    *reinterpret_cast<T2*>(&dst[dst_offset + sin_offset]) = cuda_cast<T2>(sin_temb);
    *reinterpret_cast<T2*>(&dst[dst_offset + cos_offset]) = cuda_cast<T2>(cos_temb);
}

// __global__ void TimeProjectionMulti(float*       dst,
//                                     const float* timestep,
//                                     const float* exponents,
//                                     const size_t sin_offset,
//                                     const size_t cos_offset,
//                                     const size_t half_dim_)
// {
//     // grid x 维度代表, batch_idx
//     // block x 维度为 half_dim_ / 2
//     int dst_offset = blockIdx.x * half_dim_ * 2 + threadIdx.x * 2;

//     size_t channel_idx_0     = threadIdx.x * 2;
//     size_t channel_idx_1     = threadIdx.x * 2 + 1;
//     float2 exponent_factor_2 = make_float2(exponents[channel_idx_0] * (*(timestep + blockIdx.x)),
//                                            exponents[channel_idx_1] * (*(timestep + blockIdx.x)));

//     float2 sin_temb;
//     sin_temb.x = sinf(exponent_factor_2.x);
//     sin_temb.y = sinf(exponent_factor_2.y);

//     float2 cos_temb;
//     cos_temb.x = cosf(exponent_factor_2.x);
//     cos_temb.y = cosf(exponent_factor_2.y);

//     *reinterpret_cast<float2*>(&dst[dst_offset + sin_offset]) = sin_temb;
//     *reinterpret_cast<float2*>(&dst[dst_offset + cos_offset]) = cos_temb;
// }
template<typename T>
__global__ void TimeProjectionMulti(T*           dst,
                                    const float* timestep,
                                    const float* exponents,
                                    const size_t sin_offset,
                                    const size_t cos_offset,
                                    const size_t half_dim_)
{
    // grid x 维度代表, batch_idx
    // block x 维度为 half_dim_，因为这个时候是处理一批，不是处理单个
    // half_dim = 128,hald_dim*2 = 256
    // using T2 = typename TypeConverter<T>::Type;

    int block_idx = blockIdx.x * half_dim_ * 2;
    int data_idx  = block_idx + threadIdx.x;

    float v    = exponents[threadIdx.x] * timestep[blockIdx.x];
    float sinv = sinf(v);
    float cosv = cosf(v);

    dst[data_idx + sin_offset] = cuda_cast<T>(sinv);
    dst[data_idx + cos_offset] = cuda_cast<T>(cosv);
}

template<typename T>
void invokeTimeProjection(T*           dst,
                          const float* timestep,
                          const float* exponents,
                          const bool&  flip_sin_to_cos_,
                          const size_t half_dim_,
                          const size_t batch_size,
                          cudaStream_t stream)
{
    // dim 的长度为 320, half_dim_为 160
    int threadsPerBlock = half_dim_ / 2;
    int numBlocks       = batch_size;

    size_t sin_offset;
    size_t cos_offset;

    if (flip_sin_to_cos_ == true) {
        // temb: [cos_0, ..., sin_0, ...]
        sin_offset = half_dim_;
        cos_offset = 0;
    }
    else {
        // temb: [sin_0, ..., cos_0, ...]
        sin_offset = 0;
        cos_offset = half_dim_;
    }

    // printf("numBlocks: %d, threadsPerBlock: %d\n", numBlocks, threadsPerBlock);
    TimeProjection<<<numBlocks, threadsPerBlock, 0, stream>>>(
        dst, timestep, exponents, sin_offset, cos_offset, half_dim_);
}

template<typename T>
void invokeTimeProjectionMulti(T*           dst,
                               const float* timestep,
                               int          n_timestep,
                               const float* exponents,
                               const bool&  flip_sin_to_cos_,
                               const size_t half_dim_,
                               const size_t batch_size,
                               cudaStream_t stream)
{
    // dim 的长度为 320, half_dim_为 160
    int    threadsPerBlock = half_dim_;                // 128
    int    numBlocks       = batch_size * n_timestep;  // 12
    size_t sin_offset;
    size_t cos_offset;

    if (flip_sin_to_cos_ == true) {
        // temb: [cos_0, ..., sin_0, ...]
        sin_offset = half_dim_;
        cos_offset = 0;
    }
    else {
        // temb: [sin_0, ..., cos_0, ...]
        sin_offset = 0;
        cos_offset = half_dim_;
    }

    // printf("numBlocks: %d, threadsPerBlock: %d\n", numBlocks, threadsPerBlock);
    TimeProjectionMulti<<<numBlocks, threadsPerBlock, 0, stream>>>(
        dst, timestep, exponents, sin_offset, cos_offset, half_dim_);
}

#define INSTANTIATE_INVOKE_TIME_PROJECTION_MULTI(T)                                                                    \
    template void invokeTimeProjectionMulti(T*           dst,                                                          \
                                            const float* timestep,                                                     \
                                            int          n_timestep,                                                   \
                                            const float* exponents,                                                    \
                                            const bool&  flip_sin_to_cos_,                                             \
                                            const size_t half_dim_,                                                    \
                                            const size_t batch_size,                                                   \
                                            cudaStream_t stream)

INSTANTIATE_INVOKE_TIME_PROJECTION_MULTI(float);
INSTANTIATE_INVOKE_TIME_PROJECTION_MULTI(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_TIME_PROJECTION_MULTI(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_TIME_PROJECTION_MULTI

#define INSTANTIATE_INVOKE_TIME_PROJECTION(T)                                                                          \
    template void invokeTimeProjection(T*           dst,                                                               \
                                       const float* timestep,                                                          \
                                       const float* exponents,                                                         \
                                       const bool&  flip_sin_to_cos_,                                                  \
                                       const size_t half_dim_,                                                         \
                                       const size_t batch_size,                                                        \
                                       cudaStream_t stream)

INSTANTIATE_INVOKE_TIME_PROJECTION(float);
INSTANTIATE_INVOKE_TIME_PROJECTION(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_TIME_PROJECTION(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_TIME_PROJECTION

}  // namespace lyradiff