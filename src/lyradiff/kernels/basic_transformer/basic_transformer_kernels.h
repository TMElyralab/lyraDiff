#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeSelfAttnKernelInputPermute(T*           dst,
                                      const T*     src,
                                      size_t       batch_size,
                                      size_t       seq_len,
                                      size_t       dim,
                                      size_t       head_num,
                                      size_t       dim_per_head,
                                      cudaStream_t stream);

template<typename T>
void invokeCrossAttnKernelInputPermute(T*           dst,
                                       const T*     src,
                                       size_t       batch_size,
                                       size_t       seq_len,
                                       size_t       dim,
                                       size_t       head_num,
                                       size_t       dim_per_head,
                                       cudaStream_t stream);

template<typename T>
void invokeCrossAttnKernelInputPermuteWithOffset(T*           dst,
                                                 const T*     src,
                                                 size_t       batch_size,
                                                 size_t       seq_len,
                                                 size_t       dim,
                                                 size_t       head_num,
                                                 size_t       dim_per_head,
                                                 size_t       offset,
                                                 size_t       offset2,
                                                 float        scale_k,
                                                 cudaStream_t stream);

template<typename T>
void invokeCrossAttnKernelInputPermuteAndScalePtrWithOffset(T*           dst,
                                                            const T*     src,
                                                            size_t       batch_size,
                                                            size_t       seq_len,
                                                            size_t       dim,
                                                            size_t       head_num,
                                                            size_t       dim_per_head,
                                                            size_t       offset,
                                                            size_t       offset2,
                                                            size_t       scale_len,
                                                            float*       scale_k,
                                                            cudaStream_t stream);

template<typename T>
void invokeSelfAttn2KernelInputPermute(
    T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream);

template<typename T>
void invokeCrossAttn2KernelInputPermute(
    T* dst, const T* src, size_t batch_size, size_t seq_len, size_t dim_per_group, cudaStream_t stream);

template<typename T>
void invokeCrossAttn2KernelInputPermuteWithOffset(T*           dst,
                                                  const T*     src,
                                                  size_t       batch_size,
                                                  size_t       seq_len,
                                                  size_t       dim_per_group,
                                                  size_t       offset,
                                                  size_t       total_seq_len,
                                                  float        scale_k,
                                                  cudaStream_t stream);

template<typename T>
void invokeCrossAttn2KernelInputPermuteAndScalePtrWithOffset(T*           dst,
                                                             const T*     src,
                                                             size_t       batch_size,
                                                             size_t       seq_len,
                                                             size_t       dim_per_group,
                                                             size_t       offset,
                                                             size_t       total_seq_len,
                                                             size_t       scale_len,
                                                             float*       scale_k,
                                                             cudaStream_t stream);

template<typename T>
void invokeFusedBiasResidualAdd(T*           dst,
                                const T*     src,
                                const T*     residual,
                                const T*     bias,
                                size_t       batch_size,
                                size_t       seq_len,
                                size_t       dim,
                                cudaStream_t stream);

template<typename T>
void invokeIpMaskAndAddResidual(T*           dst,
                                const T*     attention_res,
                                const T*     ip_res,
                                const T*     ip_region_mask,
                                const float  ip_scale,
                                const size_t channel,
                                const size_t seq_len,
                                const size_t batch_size,
                                cudaStream_t stream);

}  // namespace lyradiff