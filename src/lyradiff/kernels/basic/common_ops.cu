#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

// input format B * Dim

template<typename T>
__global__ void concat2d(T* dst, const T* src1, const T* src2, int batch_size, int concat_dim1, int concat_dim2)
{
    // grid: (bs,(concat1+concat2)
    // threads per block: 1, each

    size_t feature_size = concat_dim1 + concat_dim2;
    size_t rel_idx      = blockIdx.y;
    size_t dst_idx      = blockIdx.x * feature_size + rel_idx;
    size_t src1_idx     = blockIdx.x * concat_dim1 + rel_idx;
    size_t src2_idx     = blockIdx.x * concat_dim2 + rel_idx - concat_dim1;

    if (rel_idx < concat_dim1) {
        dst[dst_idx] = src1[src1_idx];
    }
    else if (rel_idx < (concat_dim1 + concat_dim2)) {
        dst[dst_idx] = src2[src2_idx];
    }
    else {
        return;
    }
}

template<typename T>
void invokeConcat2d(T* dst, const T* src1, const T* src2, int batch_size, int dim1, int dim2, cudaStream_t stream)
{
    // 目前只支持在最后一个维度concat
    dim3 grid(batch_size, dim1 + dim2, 1);
    concat2d<<<grid, 1, 0, stream>>>(dst, src1, src2, batch_size, dim1, dim2);
}

#define INSTANTIATE_INVOKE_CONCAT2D(T)                                                                                 \
    template void invokeConcat2d(                                                                                      \
        T* dst, const T* src1, const T* src2, int batch_size, int dim1, int dim2, cudaStream_t stream)

INSTANTIATE_INVOKE_CONCAT2D(float);
INSTANTIATE_INVOKE_CONCAT2D(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_CONCAT2D(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_CONCAT2D

template<typename T>
__global__ void addTensor2d(T* dst, const T* src1, const T* src2, size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = src1[tid] + src2[tid];
        // dst[tid] += src[tid];
    }
}

template<typename T>
void invokeAddTensor2d(T* dst, const T* src1, const T* src2, size_t batch_size, size_t dim, cudaStream_t stream)
{
    addTensor2d<<<256, 256, 0, stream>>>(dst, src1, src2, batch_size * dim);
}
#define INSTANTIATE_INVOKE_ADDTENSOR2D(T)                                                                              \
    template void invokeAddTensor2d(                                                                                   \
        T* dst, const T* src1, const T* src2, size_t batch_size, size_t dim, cudaStream_t stream)

INSTANTIATE_INVOKE_ADDTENSOR2D(float);
INSTANTIATE_INVOKE_ADDTENSOR2D(half);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_ADDTENSOR2D(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_ADDTENSOR2D

template<typename T>
__global__ void add3Tensor2d(T* dst, const T* src1, const T* src2, const T* src3, size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = src1[tid] + src2[tid] + src3[tid];
        // dst[tid] += src[tid];
    }
}

template<typename T>
__global__ void add3Tensor2dAndSilu(T* dst, const T* src1, const T* src2, const T* src3, size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        T d      = src1[tid] + src2[tid] + src3[tid];
        dst[tid] = cuda_cast<T>(SiluActivation<float>::apply(cuda_cast<float>(d)));
    }
}

template<typename T>
void invokeAdd3Tensor2d(
    T* dst, const T* src1, const T* src2, const T* src3, size_t batch_size, size_t dim, bool silu, cudaStream_t stream)
{
    if (silu) {
        add3Tensor2dAndSilu<<<256, 256, 0, stream>>>(dst, src1, src2, src3, batch_size * dim);
    }
    else {
        add3Tensor2d<<<256, 256, 0, stream>>>(dst, src1, src2, src3, batch_size * dim);
    }
}

template<typename T_OUT, typename T_IN>
__global__ void transpose102(T_OUT* dst, T_IN* src, const int dim0, const int dim1, const int dim2)
{
    // src permutation: [0, 1, 2]
    // dst permutation: [1, 0, 2]
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < dim0 * dim1 * dim2; tid += blockDim.x * gridDim.x) {
        int       tmp_idx                                           = tid;
        const int dim_2_idx                                         = tmp_idx % dim2;
        tmp_idx                                                     = (tmp_idx - dim_2_idx) / dim2;
        const int dim_1_idx                                         = tmp_idx % dim1;
        tmp_idx                                                     = (tmp_idx - dim_1_idx) / dim1;
        const int dim_0_idx                                         = tmp_idx % dim0;
        dst[dim_1_idx * dim0 * dim2 + dim_0_idx * dim2 + dim_2_idx] = src[tid];
    }
}

template<typename T>
void invokeTranspose102(T* dst, T* src, const int dim0, const int dim1, const int dim2)
{
    transpose102<<<256, 256>>>(dst, src, dim0, dim1, dim2);
}

#ifdef ENABLE_BF16
template void
invokeTranspose102(__nv_bfloat16* dst, __nv_bfloat16* src, const int dim0, const int dim1, const int dim2);
#endif  // ENABLE_BF16
template void invokeTranspose102(float* dst, float* src, const int dim0, const int dim1, const int dim2);
template void invokeTranspose102(half* dst, half* src, const int dim0, const int dim1, const int dim2);

#define INSTANTIATE_INVOKE_ADD3TENSOR2D(T)                                                                             \
    template void invokeAdd3Tensor2d(T*           dst,                                                                 \
                                     const T*     src1,                                                                \
                                     const T*     src2,                                                                \
                                     const T*     src3,                                                                \
                                     size_t       batch_size,                                                          \
                                     size_t       dim,                                                                 \
                                     bool         silu,                                                                \
                                     cudaStream_t stream)

INSTANTIATE_INVOKE_ADD3TENSOR2D(float);
INSTANTIATE_INVOKE_ADD3TENSOR2D(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_ADD3TENSOR2D(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_ADD3TENSOR2D

}  // namespace lyradiff
