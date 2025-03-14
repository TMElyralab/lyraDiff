#include "cuda_utils.h"
#include "src/lyradiff/layers/gemm_utils/SmoothQuantGemm.h"
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/cublasAlgoMap.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>

#pragma once

using namespace lyradiff::gemm;

namespace lyradiff {

class cublasMMWrapper {
protected:
    cublasLtHandle_t cublaslt_handle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t                               cusparselt_handle_;
    std::map<std::string, cusparseLtMatDescriptor_t> sp_mat_A_desc_map_;
    std::map<std::string, cusparseLtMatDescriptor_t> sp_mat_B_desc_map_;
    std::map<std::string, cusparseLtMatDescriptor_t> sp_mat_C_desc_map_;
#endif

    cudaDataType_t Atype_;
    cudaDataType_t Btype_;
    cudaDataType_t Ctype_;
    cudaDataType_t computeType_;

    cudaStream_t stream_;
    std::mutex*  mu_;

    IAllocator* allocator_ = nullptr;

    half* lora_workspace_ = nullptr;

    friend class cublasINT8MMWrapper;

    void _Int8Gemm(const int     m,
                   const int     n,
                   const int     k,
                   const int8_t* A,
                   const int     lda,
                   const int8_t* B,
                   const int     ldb,
                   void*         C,
                   const int     ldc,
                   const void*   alpha,
                   const int     mode,
                   const bool    per_column_scaling);

public:
    cublasAlgoMap* cublas_algo_map_;

    void* cublas_workspace_ = nullptr;

    std::shared_ptr<SmoothQuantGemmPluginProfiler>        int8_gemm_profiler_;
    std::shared_ptr<SmoothQuantGemmPlugin<half>>          fp16_int8_matmul_runner_;
    std::shared_ptr<SmoothQuantGemmPlugin<__nv_bfloat16>> bf16_int8_matmul_runner_;

    cublasHandle_t cublas_handle_;
    cublasMMWrapper(cublasHandle_t   cublas_handle_,
                    cublasLtHandle_t cublaslt_handle_,
                    cudaStream_t     stream,
                    cublasAlgoMap*   map,
                    std::mutex*      mu,
                    IAllocator*      allocator);

#ifdef SPARSITY_ENABLED
    cublasMMWrapper(cublasHandle_t     cublas_handle_,
                    cublasLtHandle_t   cublaslt_handle_,
                    cusparseLtHandle_t cusparselt_handle,
                    cudaStream_t       stream,
                    cublasAlgoMap*     map,
                    std::mutex*        mu,
                    IAllocator*        allocator);
#endif

    ~cublasMMWrapper();

    cublasMMWrapper(const cublasMMWrapper& wrapper);

    virtual void cublasVersionCheck()
    {
        return;
    };
    cublasStatus_t cublasLtMatmulWrapper(cublasLtHandle_t            lightHandle,
                                         cublasLtMatmulDesc_t        computeDesc,
                                         const void*                 alpha,
                                         const void*                 A,
                                         cublasLtMatrixLayout_t      Adesc,
                                         const void*                 B,
                                         cublasLtMatrixLayout_t      Bdesc,
                                         const void*                 beta,
                                         const void*                 C,
                                         cublasLtMatrixLayout_t      Cdesc,
                                         void*                       D,
                                         cublasLtMatrixLayout_t      Ddesc,
                                         const cublasLtMatmulAlgo_t* algo,
                                         void*                       workspace,
                                         size_t                      workspaceSizeInBytes,
                                         cudaStream_t                stream);

    int findBestAlgoFp16(cublasOperation_t        transa,
                         cublasOperation_t        transb,
                         const int                m,
                         const int                n,
                         const int                k,
                         const void*              A,
                         const int                lda,
                         const void*              B,
                         const int                ldb,
                         void*                    C,
                         const int                ldc,
                         cublasLtMatmulAlgo_info& best_algo);

    std::pair<bool, cublasLtMatmulAlgo_t> findBestAlgo(cublasLtHandle_t       lightHandle,
                                                       cublasLtMatmulDesc_t   computeDesc,
                                                       const void*            alpha,
                                                       const void*            A,
                                                       cublasLtMatrixLayout_t Adesc,
                                                       const void*            B,
                                                       cublasLtMatrixLayout_t Bdesc,
                                                       const void*            beta,
                                                       const void*            C,
                                                       cublasLtMatrixLayout_t Cdesc,
                                                       void*                  D,
                                                       cublasLtMatrixLayout_t Ddesc,
                                                       cudaStream_t           stream);

    cublasLtMatmulAlgo_t searchBestAlgo(cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        const int         m,
                                        const int         n,
                                        const int         k,
                                        const void*       A,
                                        const int         lda,
                                        const void*       B,
                                        const int         ldb,
                                        void*             C,
                                        const int         ldc);

    using MatrixLayout = std::tuple<cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
    using cache_idx_t  = std::tuple<cublasLtMatmulDesc_t, std::array<MatrixLayout, 4>>;
    std::map<cache_idx_t, cublasLtMatmulAlgo_t> algo_cache;

    MatrixLayout createMatrixLayout(cublasLtMatrixLayout_t Mdesc);

    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       alpha,
              const void*       A,
              cudaDataType_t    Atype,
              int               lda,
              const void*       B,
              cudaDataType_t    Btype,
              int               ldb,
              const void*       beta,
              void*             C,
              cudaDataType_t    Ctype,
              int               ldc,
              cudaDataType_t    computeType,
              cublasGemmAlgo_t  algo);

    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              const int         lda,
              const void*       B,
              const int         ldb,
              void*             C,
              const int         ldc);

    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              const int         lda,
              const void*       B,
              const int         ldb,
              void*             C,
              const int         ldc,
              float             f_alpha,
              float             f_beta,
              bool              has_hook = true);

    void Int8Gemm(const int     m,
                  const int     n,
                  const int     k,
                  const int8_t* A,
                  const int     lda,
                  const int8_t* B,
                  const int     ldb,
                  int8_t*       C,
                  const int     ldc,
                  const float*  alpha,
                  const bool    per_column_scaling = false);

    void Int8Gemm(const int     m,
                  const int     n,
                  const int     k,
                  const int8_t* A,
                  const int     lda,
                  const int8_t* B,
                  const int     ldb,
                  int32_t*      C,
                  const int     ldc);

    void setFP32GemmConfig();
    void setFP16GemmConfig();
#ifdef ENABLE_BF16
    void setBF16GemmConfig();
#endif
    void setStream(cudaStream_t stream);

    void setGemmConfig(cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType);

    CublasDataType getCublasDataType(cudaDataType_t data_type);

#if (CUDART_VERSION >= 11000)
    void Gemm(cublasOperation_t transa,
              cublasOperation_t transb,
              const int         m,
              const int         n,
              const int         k,
              const void*       A,
              const int         lda,
              const void*       B,
              const int         ldb,
              const void*       bias,
              void*             C,
              const int         ldc);
#endif

    void stridedBatchedGemm(cublasOperation_t transa,
                            cublasOperation_t transb,
                            const int         m,
                            const int         n,
                            const int         k,
                            const void*       A,
                            const int         lda,
                            const int64_t     strideA,
                            const void*       B,
                            const int         ldb,
                            const int64_t     strideB,
                            void*             C,
                            const int         ldc,
                            const int64_t     strideC,
                            const int         batchCount,
                            const float       f_alpha = 1.0f,
                            const float       f_beta  = 0.0f);

    void stridedBatchedGemm(cublasOperation_t transa,
                            cublasOperation_t transb,
                            const int         m,
                            const int         n,
                            const int         k,
                            const float       f_alpha,
                            const void*       A,
                            cudaDataType_t    AType,
                            const int         lda,
                            const int64_t     strideA,
                            const void*       B,
                            cudaDataType_t    BType,
                            const int         ldb,
                            const int64_t     strideB,
                            const float       f_beta,
                            void*             C,
                            cudaDataType_t    CType,
                            const int         ldc,
                            const int64_t     strideC,
                            const int         batch_count,
                            cudaDataType_t    computeType);

    void batchedGemm(cublasOperation_t  transa,
                     cublasOperation_t  transb,
                     const int          m,
                     const int          n,
                     const int          k,
                     const void* const* A,
                     const int          lda,
                     const void* const* B,
                     const int          ldb,
                     void* const*       C,
                     const int          ldc,
                     const int          batch_count);

#ifdef ENABLE_FP8
    void GemmWithResidualAndBias(cublasOperation_t    transa,
                                 cublasOperation_t    transb,
                                 const int            m,
                                 const int            n,
                                 const int            k,
                                 const __nv_fp8_e4m3* A,
                                 const int            lda,
                                 const float*         a_scale,
                                 const __nv_fp8_e4m3* B,
                                 const int            ldb,
                                 const float*         b_scale,
                                 void*                C,
                                 const int            ldc,
                                 const void*          bias     = nullptr,
                                 const void*          Residual = nullptr,
                                 float                f_alpha  = 1.0f,
                                 float                f_beta   = 0.0f,
                                 const bool           is_gelu  = false);
#endif  // ENABLE_FP8

    // kio: need hack lora
    void GemmWithResidualAndBias(cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 const int         m,
                                 const int         n,
                                 const int         k,
                                 const void*       A,
                                 const int         lda,
                                 const void*       B,
                                 const int         ldb,
                                 void*             C,
                                 const int         ldc,
                                 const void*       bias     = nullptr,
                                 const void*       Residual = nullptr,
                                 float             f_alpha  = 1.0f,
                                 float             f_beta   = 0.0f,
                                 const bool        is_gelu  = false,
                                 bool              has_hook = true);

    // for s3diff hook
    void* GemmS3DiffLoraHook(cublasOperation_t transa,
                      cublasOperation_t transb,
                      const int         m,
                      const int         n,
                      const int         k,
                      const void*       A,
                      const int         lda,
                      const void*       B,
                      const int         ldb,
                      void*             C,
                      const int         ldc);

    bool isFuseBatchGemm(const int batch_count, const int m, const int k, const int n);

    void profileGemmInt8(const int maxM, const int n, const int k)
    {
        if (Atype_ == CUDA_R_16F) {
            fp16_int8_matmul_runner_->configurePlugin(maxM, k, n);
            fp16_int8_matmul_runner_->initialize();
            cudaDeviceSynchronize();
        }
        else if (Atype_ == CUDA_R_16BF) {
            bf16_int8_matmul_runner_->configurePlugin(maxM, k, n);
            bf16_int8_matmul_runner_->initialize();
            cudaDeviceSynchronize();
        }
    };

    void runGemmInt8(int8_t const* A,
                     int8_t const* B,
                     float const*  alphaCol,
                     float const*  alphaRow,
                     void*         C,
                     int           m,
                     int           n,
                     int           k,
                     char*         workspacePtr,
                     cudaStream_t  stream)
    {
        if (Atype_ == CUDA_R_16F) {
            fp16_int8_matmul_runner_->RunGemm(A, B, alphaCol, alphaRow, C, m, n, k, workspacePtr, stream);
        }
        else if (Atype_ == CUDA_R_16BF) {
            bf16_int8_matmul_runner_->RunGemm(A, B, alphaCol, alphaRow, C, m, n, k, workspacePtr, stream);
        }
    };

#ifdef SPARSITY_ENABLED
    void SpGemm(cublasOperation_t transa,
                cublasOperation_t transb,
                const int         m,
                const int         n,
                const int         k,
                const void*       A,
                const void*       B,
                void*             C);

    size_t getSparseMatrixSize(int m, int k);
    void   compressMatrix(const void* input, void* output, const int m, const int k);

    bool isUseSparse(const int batch_count, const int m, const int n, const int k);
#endif
};

extern cublasMMWrapper* cublas_wrapper_glob;
}  // namespace lyradiff
