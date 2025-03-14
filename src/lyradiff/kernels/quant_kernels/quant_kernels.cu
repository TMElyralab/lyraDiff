#include "int8_utils.cuh"
#include "quant_kernels.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"

namespace lyradiff {

constexpr int kWarpSize = 32;

template<int thread_group_width = kWarpSize>
__inline__ __device__ void AmaxWarpReduce(float thread_amax, float* amax)
{
    *amax = thread_amax;

    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        float b_amax = __shfl_down_sync(0xffffffff, *amax, mask, thread_group_width);
        *amax        = max(*amax, b_amax);
    }
}

__inline__ __device__ void AmaxBlockAllReduce(float thread_amax, float* result_amax)
{
    __shared__ float amax_shared[kWarpSize];
    __shared__ float amax_result_broadcast;
    const int        lid       = threadIdx.x % kWarpSize;
    const int        wid       = threadIdx.x / kWarpSize;
    float            warp_amax = 0;
    AmaxWarpReduce(thread_amax, &warp_amax);
    __syncthreads();
    if (lid == 0) {
        amax_shared[wid] = warp_amax;
    }
    __syncthreads();
    if (wid == 0) {
        if (threadIdx.x < blockDim.x / kWarpSize) {
            warp_amax = amax_shared[lid];
        }
        else {
            warp_amax = static_cast<float>(0);
        }
        __syncwarp();
        float block_amax = 0;
        AmaxWarpReduce(warp_amax, &block_amax);
        if (lid == 0) {
            amax_result_broadcast = block_amax;
        }
    }
    __syncthreads();
    *result_amax = amax_result_broadcast;
}

template<typename T>
__global__ void sqInputQuant(int8_t*      dst,
                             const T*     src,
                             const float* pre_quant_scale,
                             const float* input_quant_scale,
                             size_t       batch_size,
                             size_t       seq_len,
                             size_t       dim)
{
    int cur_row = blockIdx.x;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // T2     src_2                 = __ldg((const T2*)(src + cur_row * dim + cur_col));
    // float2 pre_quant_scale_2     = __ldg((const float2*)(pre_quant_scale + cur_col));
    // float  cur_input_quant_scale = __ldg(input_quant_scale);

    // dst[cur_row * dim + cur_col] =
    //     float_to_int8_rn(cuda_cast<float>(src_2.x) * pre_quant_scale_2.x / cur_input_quant_scale);
    // dst[cur_row * dim + cur_col + 1] =
    //     float_to_int8_rn(cuda_cast<float>(src_2.y) * pre_quant_scale_2.y / cur_input_quant_scale);

    for (uint idx = threadIdx.x; idx < dim / 2; idx += blockDim.x) {
        // float2 tmp2 = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
        // float2 sca2                   = cuda_cast<float2>(scale_ptr[idx]);
        // tmp2.x                        = tmp2.x * s_inv_mean * sca2.x;
        // tmp2.y                        = tmp2.y * s_inv_mean * sca2.y;
        // out_ptr[blockIdx.x * n + idx] = cuda_cast<T2>(tmp2);

        int cur_col = idx * 2;

        T2     src_2                 = __ldg((const T2*)(src + cur_row * dim + cur_col));
        float2 pre_quant_scale_2     = __ldg((const float2*)(pre_quant_scale + cur_col));
        float  cur_input_quant_scale = __ldg(input_quant_scale);

        dst[cur_row * dim + cur_col] =
            float_to_int8_rn(cuda_cast<float>(src_2.x) * pre_quant_scale_2.x / cur_input_quant_scale);
        dst[cur_row * dim + cur_col + 1] =
            float_to_int8_rn(cuda_cast<float>(src_2.y) * pre_quant_scale_2.y / cur_input_quant_scale);
    }
}

template<typename T>
__global__ void sqPerTokenInputQuant(int8_t*      dst,
                                     const T*     src,
                                     const float* pre_quant_scale,
                                     const float* input_quant_scale,
                                     size_t       batch_size,
                                     size_t       seq_len,
                                     size_t       dim)
{
    int cur_row = blockIdx.x;
    int cur_col = threadIdx.x * 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    T2     src_2                 = __ldg((const T2*)(src + cur_row * dim + cur_col));
    float2 pre_quant_scale_2     = __ldg((const float2*)(pre_quant_scale + cur_col));
    float  cur_input_quant_scale = __ldg(input_quant_scale + cur_row);

    dst[cur_row * dim + cur_col] =
        float_to_int8_rn(cuda_cast<float>(src_2.x) * pre_quant_scale_2.x / cur_input_quant_scale);
    dst[cur_row * dim + cur_col + 1] =
        float_to_int8_rn(cuda_cast<float>(src_2.y) * pre_quant_scale_2.y / cur_input_quant_scale);
}

template<typename T>
__global__ void sqOutputDequant(T*             dst,
                                const int32_t* src,
                                const float*   input_quant_scale,
                                const float*   weight_quant_scale,
                                size_t         batch_size,
                                size_t         seq_len,
                                size_t         dim)
{
    int col_batch_idx = blockIdx.y;
    int block_size    = blockDim.x;
    int cur_row       = blockIdx.x;
    // int cur_col       = threadIdx.x * 2;

    int cur_col = threadIdx.x * 2 + col_batch_idx * block_size * 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    T2     src_2                 = __ldg((const T2*)(src + cur_row * dim + cur_col));
    float2 weight_quant_scale_2  = __ldg((const float2*)(weight_quant_scale + cur_col));
    float  cur_input_quant_scale = __ldg(input_quant_scale);

    dst[cur_row * dim + cur_col] =
        cuda_cast<T>(__int2float_rn(src_2.x) * cur_input_quant_scale * weight_quant_scale_2.x);
    dst[cur_row * dim + cur_col + 1] =
        cuda_cast<T>(__int2float_rn(src_2.y) * cur_input_quant_scale * weight_quant_scale_2.y);
}

template<typename T>
__global__ void sqPerTokenOutputDequant(T*             dst,
                                        const int32_t* src,
                                        const float*   input_quant_scale,
                                        const float*   weight_quant_scale,
                                        size_t         batch_size,
                                        size_t         seq_len,
                                        size_t         dim)
{
    int col_batch_idx = blockIdx.y;
    int block_size    = blockDim.x;
    int cur_row       = blockIdx.x;
    // int cur_col       = threadIdx.x * 2;

    int cur_col = threadIdx.x * 2 + col_batch_idx * block_size * 2;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    T2     src_2                 = __ldg((const T2*)(src + cur_row * dim + cur_col));
    float2 weight_quant_scale_2  = __ldg((const float2*)(weight_quant_scale + cur_col));
    float  cur_input_quant_scale = __ldg(input_quant_scale + cur_row);

    dst[cur_row * dim + cur_col] =
        cuda_cast<T>(__int2float_rn(src_2.x) * cur_input_quant_scale * weight_quant_scale_2.x);
    dst[cur_row * dim + cur_col + 1] =
        cuda_cast<T>(__int2float_rn(src_2.y) * cur_input_quant_scale * weight_quant_scale_2.y);
}

template<typename T, int32_t buffer_size>
__global__ void GetINT8WeightScale(float*       new_weight_scale,
                                   int8_t*      quantized_weight,
                                   const T*     weight,
                                   const float* pre_quant_scale,
                                   const size_t d_out,
                                   const size_t d_in,
                                   const size_t chunk_size)
{
    int row_id = blockIdx.x;
    int tid    = threadIdx.x;

    float                          thread_amax = 0.0f;
    __align__(sizeof(float)) float weight_buf[buffer_size];

    // step1
    for (int i = 0; i < chunk_size; i++) {
        int32_t offset    = row_id * d_in + tid * chunk_size + i;
        int32_t gb_offset = tid * chunk_size + i;

        if (offset >= d_out * d_in || gb_offset >= d_in) {
            break;
        }

        float cur_pre_quant_scale = __ldg(pre_quant_scale + gb_offset);
        weight_buf[i]             = static_cast<float>(weight[offset]) / cur_pre_quant_scale;
        thread_amax               = max(thread_amax, fabsf(weight_buf[i]));
    }

    // step2
    float amax = 0.0f;
    AmaxBlockAllReduce(thread_amax, &amax);

    // step3
    float cur_weight_scale = amax / 127.0;
    if (tid == 0) {
        new_weight_scale[row_id] = cur_weight_scale;
    }

    // step4
    for (int i = 0; i < chunk_size; i++) {
        int32_t offset    = row_id * d_in + tid * chunk_size + i;
        int32_t gb_offset = tid * chunk_size + i;

        if (offset >= d_out * d_in || gb_offset >= d_in) {
            break;
        }

        quantized_weight[offset] = float_to_int8_rn(weight_buf[i] / cur_weight_scale);
    }
}

template<typename T>
void invokeSqInputQuant(int8_t*      dst,
                        const T*     src,
                        const float* pre_quant_scale,
                        const float* input_quant_scale,
                        QuantMode    quant_mode,
                        size_t       batch_size,
                        size_t       seq_len,
                        size_t       dim,
                        cudaStream_t stream)
{
    int d = dim / 2;
    dim3 block(std::min(d, 1024));
    dim3 grid(seq_len * batch_size);

    if (quant_mode.hasPerTokenScaling()) {
        sqPerTokenInputQuant<T>
            <<<grid, block, 0, stream>>>(dst, src, pre_quant_scale, input_quant_scale, batch_size, seq_len, dim);
    }
    else {
        sqInputQuant<T>
            <<<grid, block, 0, stream>>>(dst, src, pre_quant_scale, input_quant_scale, batch_size, seq_len, dim);
    }
}

template<typename T>
void invokeSqOutputDequant(T*             dst,
                           const int32_t* src,
                           const float*   input_quant_scale,
                           const float*   weight_quant_scale,
                           QuantMode      quant_mode,
                           size_t         batch_size,
                           size_t         seq_len,
                           size_t         dim,
                           cudaStream_t   stream)
{
    size_t block_size = 128;

    dim3 grid(seq_len * batch_size, dim / block_size / 2);
    dim3 block(block_size);

    sqOutputDequant<T>
        <<<grid, block, 0, stream>>>(dst, src, input_quant_scale, weight_quant_scale, batch_size, seq_len, dim);
}

template<typename T>
void invokeGetINT8WeightScale(float*       new_weight_scale,
                              int8_t*      quantized_weight,
                              const T*     weight,
                              const float* pre_quant_scale,
                              const size_t d_out,
                              const size_t d_in,
                              cudaStream_t stream)
{

    size_t block_size = 128;
    size_t grid(d_out);
    size_t chunk_size = (d_in - 1) / 128 + 1;

    if (chunk_size <= 5) {
        GetINT8WeightScale<T, 5><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else if (chunk_size <= 10) {
        GetINT8WeightScale<T, 10><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else if (chunk_size <= 20) {
        GetINT8WeightScale<T, 20><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else if (chunk_size <= 40) {
        GetINT8WeightScale<T, 40><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else if (chunk_size <= 80) {
        GetINT8WeightScale<T, 80><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else if (chunk_size <= 160) {
        GetINT8WeightScale<T, 160><<<grid, block_size, 0, stream>>>(
            new_weight_scale, quantized_weight, weight, pre_quant_scale, d_out, d_in, chunk_size);
    }
    else {  // throw exception
        // loadInt8Lora<T, 320><<<grid, block_size, 0, stream>>>(
        //     int8_weight, weight_scale, pre_quant_scale, lora_weight, dim1, dim2, alpha, chunk_size);
    }
}

#define INSTANTIATE_INVOKE_SQ_INPUT_QUANT(T)                                                                           \
    template void invokeSqInputQuant(int8_t*      dst,                                                                 \
                                     const T*     src,                                                                 \
                                     const float* pre_quant_scale,                                                     \
                                     const float* input_quant_scale,                                                   \
                                     QuantMode    quant_mode,                                                          \
                                     size_t       batch_size,                                                          \
                                     size_t       seq_len,                                                             \
                                     size_t       dim,                                                                 \
                                     cudaStream_t stream)

INSTANTIATE_INVOKE_SQ_INPUT_QUANT(float);
INSTANTIATE_INVOKE_SQ_INPUT_QUANT(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SQ_INPUT_QUANT(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_SQ_INPUT_QUANT

#define INSTANTIATE_INVOKE_SQ_OUTPUT_DEQUANT(T)                                                                        \
    template void invokeSqOutputDequant(T*             dst,                                                            \
                                        const int32_t* src,                                                            \
                                        const float*   input_quant_scale,                                              \
                                        const float*   weight_quant_scale,                                             \
                                        QuantMode      quant_mode,                                                     \
                                        size_t         batch_size,                                                     \
                                        size_t         seq_len,                                                        \
                                        size_t         dim,                                                            \
                                        cudaStream_t   stream)

INSTANTIATE_INVOKE_SQ_OUTPUT_DEQUANT(float);
INSTANTIATE_INVOKE_SQ_OUTPUT_DEQUANT(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SQ_OUTPUT_DEQUANT(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_SQ_OUTPUT_DEQUANT

#define INSTANTIATE_INVOKE_GET_INT8_WEIGHT_SCALE(T)                                                                    \
    template void invokeGetINT8WeightScale(float*       new_weight_scale,                                              \
                                           int8_t*      quantized_weight,                                              \
                                           const T*     weight,                                                        \
                                           const float* pre_quant_scale,                                               \
                                           const size_t d_out,                                                         \
                                           const size_t d_in,                                                          \
                                           cudaStream_t stream)

INSTANTIATE_INVOKE_GET_INT8_WEIGHT_SCALE(float);
INSTANTIATE_INVOKE_GET_INT8_WEIGHT_SCALE(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_GET_INT8_WEIGHT_SCALE(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_GET_INT8_WEIGHT_SCALE

}  // namespace lyradiff
