#include "src/lyradiff/kernels/basic/gemm.h"

namespace lyradiff {
void dense_layer_kernel_launcher(const float*   in,
                                 const float*   weight,
                                 float*         out,
                                 const int      M,
                                 const int      K,
                                 const int      N,
                                 cublasHandle_t cublas_handle,
                                 cudaStream_t   stream,
                                 int            cublasAlgo)
{
    const float alpha = 1.0f, beta = 0.0f;
    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  N,
                                  M,
                                  K,
                                  &alpha,
                                  weight,
                                  CUDA_R_32F,
                                  N,
                                  in,
                                  CUDA_R_32F,
                                  K,
                                  &beta,
                                  out,
                                  CUDA_R_32F,
                                  N,
                                  CUDA_R_32F,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}

void dense_layer_kernel_launcher(const __half*  in,
                                 const __half*  weight,
                                 __half*        out,
                                 const int      M,
                                 const int      K,
                                 const int      N,
                                 cublasHandle_t cublas_handle,
                                 cudaStream_t   stream,
                                 int            cublasAlgo)
{
    const __half alpha = (__half)1.0f, beta = (__half)0.0f;
    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  N,
                                  M,
                                  K,
                                  &alpha,
                                  weight,
                                  CUDA_R_16F,
                                  N,
                                  in,
                                  CUDA_R_16F,
                                  K,
                                  &beta,
                                  out,
                                  CUDA_R_16F,
                                  N,
                                  CUDA_R_16F,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}

void cublas_Gemm_Strided_Batched(const float*      A,
                                 const float*      B,
                                 float*            out,
                                 const int         M,
                                 const int         K,
                                 const int         N,
                                 const int         batch_count,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B,
                                 float             alpha,
                                 float             beta,
                                 cublasHandle_t    cublas_handle,
                                 cudaStream_t      stream,
                                 int               cublasAlgo)
{
    const int lda = (trans_A == CUBLAS_OP_N) ? K : M;
    const int ldb = (trans_B == CUBLAS_OP_N) ? N : K;

    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                                                trans_B,
                                                trans_A,
                                                N,
                                                M,
                                                K,
                                                &alpha,
                                                B,
                                                CUDA_R_32F,
                                                ldb,
                                                K * N,
                                                A,
                                                CUDA_R_32F,
                                                lda,
                                                M * K,
                                                &beta,
                                                out,
                                                CUDA_R_32F,
                                                N,
                                                M * N,
                                                batch_count,
                                                CUDA_R_32F,
                                                static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}

void cublas_Gemm_Strided_Batched(const __half*     A,
                                 const __half*     B,
                                 __half*           out,
                                 const int         M,
                                 const int         K,
                                 const int         N,
                                 const int         batch_count,
                                 cublasOperation_t trans_A,
                                 cublasOperation_t trans_B,
                                 __half            alpha,
                                 __half            beta,
                                 cublasHandle_t    cublas_handle,
                                 cudaStream_t      stream,
                                 int               cublasAlgo)
{
    const int lda = (trans_A == CUBLAS_OP_N) ? K : M;
    const int ldb = (trans_B == CUBLAS_OP_N) ? N : K;

    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                                                trans_B,
                                                trans_A,
                                                N,
                                                M,
                                                K,
                                                &alpha,
                                                B,
                                                CUDA_R_16F,
                                                ldb,
                                                K * N,
                                                A,
                                                CUDA_R_16F,
                                                lda,
                                                M * K,
                                                &beta,
                                                out,
                                                CUDA_R_16F,
                                                N,
                                                M * N,
                                                batch_count,
                                                CUDA_R_16F,
                                                static_cast<cublasGemmAlgo_t>(cublasAlgo)));
}
}  // namespace lyradiff