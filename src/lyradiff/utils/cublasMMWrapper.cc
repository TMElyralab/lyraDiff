#include "cublasMMWrapper.h"
#include "cuda_utils.h"
#include "src/lyradiff/utils/context.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/kernels/controlnet/residual.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace lyradiff {

cublasMMWrapper* cublas_wrapper_glob = nullptr;

static inline bool time_compare(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static inline cublasStatus_t
perfMatmulRun(cublasLtHandle_t            cublaslt_handle_,  // to get the capabilities (required a GPU)
              cublasLtMatmulDesc_t        operationDesc,
              const void*                 alpha, /* host or device pointer */
              const void*                 A,
              cublasLtMatrixLayout_t      Adesc,
              const void*                 B,
              cublasLtMatrixLayout_t      Bdesc,
              const void*                 beta, /* host or device pointer */
              const void*                 C,
              cublasLtMatrixLayout_t      Cdesc,
              void*                       D,
              cublasLtMatrixLayout_t      Ddesc,
              const cublasLtMatmulAlgo_t& algo,
              int                         kernelRepeats,
              void*                       workSpace,
              size_t                      workSpaceSizeInBytes,
              customMatmulPerf_t&         perfResults,
              cudaStream_t                stream,
              cudaEvent_t&                startEvent,
              cudaEvent_t&                stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int            repeats = kernelRepeats;
    cublasStatus_t algoStatus =
        cublasLtMatmulAlgoCheck(cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cublasStatus_t oneRunStatus;
            float          time;

            oneRunStatus = cublasLtMatmul(cublaslt_handle_,
                                          operationDesc,
                                          alpha,
                                          A,
                                          Adesc,
                                          B,
                                          Bdesc,
                                          beta,
                                          C,
                                          Cdesc,
                                          D,
                                          Ddesc,
                                          &algo,
                                          workSpace,
                                          workSpaceSizeInBytes,
                                          stream);
            cudaEventRecord(startEvent, stream);
            for (int loop = 1; loop < repeats; loop++) {
                oneRunStatus = cublasLtMatmul(cublaslt_handle_,
                                              operationDesc,
                                              alpha,
                                              A,
                                              Adesc,
                                              B,
                                              Bdesc,
                                              beta,
                                              C,
                                              Cdesc,
                                              D,
                                              Ddesc,
                                              &algo,
                                              workSpace,
                                              workSpaceSizeInBytes,
                                              stream);
            }
            cudaEventRecord(stopEvent, stream);
            cudaEventSynchronize(startEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&time, startEvent, stopEvent);
            time       = time / (repeats - 1);
            algoStatus = oneRunStatus;
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                perfResults.algo          = algo;
                perfResults.time          = time;
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount    = heurResult.wavesCount;
            }
        }
        else {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
        }
    }

    return algoStatus;
}

const char* const matmulTileName[] = {
    "UNDEF",  "8x8",    "8x16",    "16x8",   "8x32",   "16x16",   "32x8",    "8x64",   "16x32",
    "32x16",  "64x8",   "32x32",   "32x64",  "64x32",  "32x128",  "64x64",   "128x32", "64x128",
    "128x64", "64x256", "128x128", "256x64", "64x512", "128x256", "256x128", "512x64",
};

void printPerfStructure(
    int m, int n, int k, const customMatmulPerf_t& perf, cublasLtMatmulAlgo_info& best_algo, int algoI)
{
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

    uint16_t inner_shape, cluster_shape;

    const cublasLtMatmulAlgo_t* matmulAlgo = &perf.algo;

    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);

    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner_shape, sizeof(inner_shape), NULL);

    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_shape, sizeof(cluster_shape), NULL);

    // memset(tmp_cstr, 0, sizeof tmp_cstr);
    // printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d "
    //        "stages=%d} status %d "
    //        "time %fms workspace=%d mathMode=%d waves=%f inner_shape=%d cluster_shape=%d\n",
    //        algoId,
    //        tile,
    //        matmulTileName[tile],
    //        numSplitsK,
    //        reductionScheme,
    //        swizzle,
    //        customOption,
    //        stages,
    //        perf.status,
    //        perf.time,
    //        (int)perf.workspaceSize,
    //        (int)perf.mathMode,
    //        perf.wavesCount,
    //        inner_shape,
    //        cluster_shape);
    // tmp_str = string(tmp_cstr);
    // gemm_test_result += tmp_str;
    if (algoI == 0) {
        best_algo.algoId          = algoId;
        best_algo.exec_time       = perf.time;
        best_algo.customOption    = customOption;
        best_algo.tile            = tile;
        best_algo.splitK_val      = numSplitsK;
        best_algo.swizzle         = swizzle;
        best_algo.reductionScheme = reductionScheme;
        best_algo.stages          = stages;
        best_algo.workspaceSize   = (int)perf.workspaceSize;
    }
}

cublasMMWrapper::cublasMMWrapper(cublasHandle_t   cublas_handle,
                                 cublasLtHandle_t cublaslt_handle,
                                 cudaStream_t     stream,
                                 cublasAlgoMap*   cublas_algo_map,
                                 std::mutex*      mu,
                                 IAllocator*      allocator):
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle),
    stream_(stream),
    cublas_algo_map_(cublas_algo_map),
    mu_(mu),
    allocator_(allocator)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
    }

    int8_gemm_profiler_ = std::make_shared<SmoothQuantGemmPluginProfiler>();
    fp16_int8_matmul_runner_ =
        std::make_shared<SmoothQuantGemmPlugin<half>>(QuantMode::useSmoothQuant(false, true), int8_gemm_profiler_);

    bf16_int8_matmul_runner_ = std::make_shared<SmoothQuantGemmPlugin<__nv_bfloat16>>(
        QuantMode::useSmoothQuant(false, true), int8_gemm_profiler_);
}

#ifdef SPARSITY_ENABLED
cublasMMWrapper::cublasMMWrapper(cublasHandle_t     cublas_handle,
                                 cublasLtHandle_t   cublaslt_handle,
                                 cusparseLtHandle_t cusparselt_handle,
                                 cudaStream_t       stream,
                                 cublasAlgoMap*     cublas_algo_map,
                                 std::mutex*        mu,
                                 IAllocator*        allocator):
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle),
    cusparselt_handle_(cusparselt_handle),
    stream_(stream),
    cublas_algo_map_(cublas_algo_map),
    mu_(mu),
    allocator_(allocator)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
    }
}
#endif

cublasMMWrapper::~cublasMMWrapper()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_ = nullptr;
    if (allocator_ != nullptr) {
        allocator_->free((void**)(&cublas_workspace_));
        allocator_ = nullptr;
    }
}

cublasMMWrapper::cublasMMWrapper(const cublasMMWrapper& wrapper):
    cublas_handle_(wrapper.cublas_handle_),
    cublaslt_handle_(wrapper.cublaslt_handle_),
#ifdef SPARSITY_ENABLED
    cusparselt_handle_(wrapper.cusparselt_handle_),
#endif
    stream_(wrapper.stream_),
    cublas_algo_map_(wrapper.cublas_algo_map_),
    mu_(wrapper.mu_),
    allocator_(wrapper.allocator_),
    int8_gemm_profiler_(wrapper.int8_gemm_profiler_),
    fp16_int8_matmul_runner_(wrapper.fp16_int8_matmul_runner_),
    bf16_int8_matmul_runner_(wrapper.bf16_int8_matmul_runner_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (allocator_ != nullptr) {
        cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
    }
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           cublasGemmAlgo_t  algo)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_->lock();
    check_cuda_error(cublasGemmEx(cublas_handle_,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  Atype,
                                  lda,
                                  B,
                                  Btype,
                                  ldb,
                                  beta,
                                  C,
                                  Ctype,
                                  ldc,
                                  computeType,
                                  algo));
    sync_check_cuda_error();
    mu_->unlock();
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       A,
                           const int         lda,
                           const void*       B,
                           const int         ldb,
                           void*             C,
                           const int         ldc)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void* cublasMMWrapper::GemmS3DiffLoraHook(cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          const int         m,
                                          const int         n,
                                          const int         k,
                                          const void*       A,
                                          const int         lda,
                                          const void*       B,
                                          const int         ldb,
                                          void*             C,
                                          const int         ldc)
{
    PRINTF_S3DIFF(">>>> GemmS3DiffLoraHook\n");
    // 如果有hook
    // result = Gx + scale(BAxM)     xM -> AxM -> BAxM
    // auto weight_name = weight_loader_manager_glob->map_weights_reverse[]
    auto  ctx                    = weight_loader_manager_glob->ctx;
    auto& map_module_rev_weights = weight_loader_manager_glob->map_module_rev_weights;

    if(ctx == nullptr) return nullptr;
    if (map_module_rev_weights.find(ctx->cur_running_module) == map_module_rev_weights.end()) {
        PRINTF_S3DIFF("gemm hook no find running module %s\n", ctx->cur_running_module.c_str());
        return nullptr;
    }
    PRINTF_S3DIFF("map_module_rev_weights size: %d\n", map_module_rev_weights.size());
    auto& rev_weight_map = map_module_rev_weights.at(ctx->cur_running_module);
    PRINTF_S3DIFF("rev_weight_map size: %d\n", rev_weight_map.size());
    if (rev_weight_map.find((void*)A) == rev_weight_map.end()) {
        PRINTF_S3DIFF("gemm hook get weight failed, sz: %d\n", rev_weight_map.size());
        return nullptr;
    }

    auto weight_name  = rev_weight_map.at((void*)A);
    auto lora_container = weight_loader_manager_glob->map_lora_container[ctx->cur_running_module];
    auto weight_alpha = lora_container.get_lora_weight(weight_name, true);
    auto weight_beta  = lora_container.get_lora_weight(weight_name, false);
    auto de_mod       = lora_container.get_de_mod(weight_name);
    if (weight_alpha == nullptr || weight_beta == nullptr || de_mod == nullptr) {
        PRINTF_S3DIFF("gemm hook get lora failed: %s, %d %d %d\n",
               weight_name.c_str(),
               weight_alpha == nullptr,
               weight_beta == nullptr,
               de_mod == nullptr);
        return nullptr;
    }
    PRINTF_S3DIFF("gemm hook ok %s\n", weight_name.c_str());
    int ratio = 1;
    if(weight_name.find("attn1.to_q") != weight_name.npos) ratio = 3;
    else if(weight_name.find("attentions.0.to_q") != weight_name.npos) ratio = 3;
    else if(weight_name.find("attn2.to_k") != weight_name.npos) ratio = 2;
    
    int   rank      = ctx->cur_running_module == "unet" ? 32 : 16;
    rank = rank*ratio;

    size_t   workspace_size = n * (std::max(rank, m) + rank) * 2 * 2;
    lora_workspace_ = (half*)allocator_->reMalloc(lora_workspace_, workspace_size, false);
    half* buffer1 = lora_workspace_;
    half* buffer2 = lora_workspace_ + std::max(rank, m) * n * 2;

    int mm        = rank;  // rank;
    int nn        = n;
    int kk        = k;

    PRINTF_S3DIFF("gemm name: %s, rank: %d [%d %d %d] [%d %d %d]\n", weight_name.c_str(),rank, m,n,k,mm,nn,kk);
    PRINTF_S3DIFF("mm nn kk: [%d %d %d]\n", mm,nn,kk);
    static int lora_fwd_cnt = 0;
    lora_fwd_cnt+=1;
    PRINTF_S3DIFF("lora_fwd_cnt: %d\n", lora_fwd_cnt);

    // TODO: 计算优化：调整 GEMM 顺序
    // [m,k] [k,r] -> [m,r]
    this->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, mm, nn, kk, weight_alpha, kk, B, kk, buffer1, mm, 1.0f, 0.0f, false);

    // [m,r] [r,r] -> [m,r]
    kk = rank;
    this->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, mm, nn, kk, de_mod, kk, buffer1, kk, buffer2, mm, 1.0f, 0.0f, false);

    // [m,r] [r,n] -> [m,n]
    mm = m;
    this->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, mm, nn, kk, weight_beta, kk, buffer2, kk, buffer1, mm, 1.0f, 0.0f, false);
    return buffer1;
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           bool              has_hook)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    void* D = C;
    void* hook_C = this->GemmS3DiffLoraHook(transa, transb, m, n, k, A, lda, B, ldb, C, ldc);
    if (hook_C != nullptr) {
        // 问题：这里默认 f_beta=0，如果 f_beta > 0, 需要加残差
        PRINTF_S3DIFF("hook add residual\n");
        C      = hook_C;
        f_beta = 1.0;
    }

    // printf("current gemm m: %d, n: %d, k: %d \n", m, n, k);

    mu_->lock();
    // TODO: default cublas libs
    int  is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    bool using_cublasLt      = (Atype_ == CUDA_R_16F) ? true : false;
    int  batch_count         = 1;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    // cublas_algo_map_->insertShape(batch_count, m, n, k);

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));
    if (findAlgo) {
        if (info.stages != -1) {
            using_cublasLt = true;
        }
        else {
            using_cublasLt = false;
        }
    }

    if (using_cublasLt) {
        cublasLtMatmulDesc_t   operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        cudaDataType_t         scaleType;
#if (CUDART_VERSION >= 11000)
        cublasComputeType_t computeType;
#else
        cudaDataType_t computeType;
#endif

        if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_16F;
#else
            computeType = CUDA_R_16F;
#endif
            scaleType = CUDA_R_16F;
        }
        else {
#if (CUDART_VERSION >= 11000)
            computeType = CUBLAS_COMPUTE_32F;
#else
            computeType = CUDA_R_32F;
#endif
            scaleType = CUDA_R_32F;
        }

        // --------------------------------------
        // Create descriptors for the original matrices
        cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
        cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
        cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
        cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

        cublasLtMatmulAlgo_t algo;
        void*                workSpace     = cublas_workspace_;
        int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
        if (findAlgo) {
            if (info.workspaceSize > workspaceSize) {
                findAlgo = 0;
            }
            else {
                cublasLtMatmulAlgoInit(
                    cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, info.algoId, &algo);
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                     CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                     &(info.reductionScheme),
                                                     sizeof(info.reductionScheme));

                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));

                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));
                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                     CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                                     &(info.cluster_shapeId),
                                                     sizeof(info.cluster_shapeId));
            }
        }
        // std::cout << "cublasLtMatmul" << std::endl;
        // cudaDeviceSynchronize();
        cublasStatus_t status = cublasLtMatmul(cublaslt_handle_,
                                               operationDesc,
                                               alpha,
                                               A,
                                               Adesc,
                                               B,
                                               Bdesc,
                                               beta,
                                               C,
                                               Cdesc,
                                               D,
                                               Cdesc,
                                               (findAlgo == 1 ? (&algo) : NULL),
                                               workSpace,
                                               workspaceSize,
                                               stream_);

        if (status != CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmul(cublaslt_handle_,
                                    operationDesc,
                                    alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    beta,
                                    C,
                                    Cdesc,
                                    D,
                                    Cdesc,
                                    NULL,
                                    workSpace,
                                    workspaceSize,
                                    stream_);
        }

        // cudaDeviceSynchronize();

        cublasLtMatmulDescDestroy(operationDesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        sync_check_cuda_error();
    }
    else {
        int cublasAlgo = info.algoId;
        // std::cout << "cublasGemmEx" << std::endl;
        // cudaDeviceSynchronize();
        check_cuda_error(cublasGemmEx(cublas_handle_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      Atype_,
                                      lda,
                                      B,
                                      Btype_,
                                      ldb,
                                      beta,
                                      C,
                                      Ctype_,
                                      ldc,
                                      computeType_,
                                      static_cast<cublasGemmAlgo_t>(cublasAlgo)));
        // cudaDeviceSynchronize();
        sync_check_cuda_error();
    }
    mu_->unlock();

    // void* hook_C = nullptr;

    // if (has_hook) {
    //     hook_C = this->GemmS3DiffLoraHook(transa, transb, m, n, k, A, lda, B, ldb, C, ldc);
    //     if(hook_C != nullptr)
    //     invokeAddResidual((half*)C, (half*)C, (half*)hook_C, 1, m, n,1,1, stream_);
    // }

}

void cublasMMWrapper::setFP32GemmConfig()
{
    Atype_       = CUDA_R_32F;
    Btype_       = CUDA_R_32F;
    Ctype_       = CUDA_R_32F;
    computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::setFP16GemmConfig()
{
    Atype_ = CUDA_R_16F;
    Btype_ = CUDA_R_16F;
    Ctype_ = CUDA_R_16F;
    // computeType_ = CUDA_R_16F;
    computeType_ = CUDA_R_32F;
}

#ifdef ENABLE_BF16
void cublasMMWrapper::setBF16GemmConfig()
{
    Atype_       = CUDA_R_16BF;
    Btype_       = CUDA_R_16BF;
    Ctype_       = CUDA_R_16BF;
    computeType_ = CUDA_R_32F;
}
#endif

void cublasMMWrapper::setGemmConfig(cudaDataType_t aType,
                                    cudaDataType_t bType,
                                    cudaDataType_t cType,
                                    cudaDataType_t computeType)
{
    Atype_       = aType;
    Btype_       = bType;
    Ctype_       = cType;
    computeType_ = computeType;
}

CublasDataType cublasMMWrapper::getCublasDataType(cudaDataType_t data_type)
{
    if (data_type == CUDA_R_16F) {
        return HALF_DATATYPE;
    }
    else if (data_type == CUDA_R_32F) {
        return FLOAT_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (data_type == CUDA_R_16BF) {
        return BFLOAT16_DATATYPE;
    }
#endif
    return FLOAT_DATATYPE;
}

#if (CUDART_VERSION >= 11000)
// input, weight, output are row-major
// only works for cublas 11.x
void cublasMMWrapper::Gemm(cublasOperation_t transa,
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
                           const int         ldc)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cudaDataType_t      Atype, Btype, Ctype;
    cublasComputeType_t computeType;
    cudaDataType_t      scaleType;
    float               alpha_float = 1.0f;
    float               beta_float  = 0.0f;
    half                alpha_half  = half(1.0f);
    half                beta_half   = half(0.0f);
    void *              alpha, *beta;

    // int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    if (Atype_ == CUDA_R_32F) {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype       = CUDA_R_32F;
        Btype       = CUDA_R_32F;
        Ctype       = CUDA_R_32F;
        scaleType   = CUDA_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    }
    else if (Atype_ == CUDA_R_16BF) {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        Atype       = CUDA_R_16BF;
        Btype       = CUDA_R_16BF;
        Ctype       = CUDA_R_16BF;
        scaleType   = CUDA_R_32F;
        alpha       = &alpha_float;
        beta        = &beta_float;
    }
    else {
        computeType = CUBLAS_COMPUTE_16F;
        Atype       = CUDA_R_16F;
        Btype       = CUDA_R_16F;
        Ctype       = CUDA_R_16F;
        scaleType   = CUDA_R_16F;
        alpha       = &alpha_half;
        beta        = &beta_half;
    }

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtEpilogue_t     epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatrixLayoutCreate(&Adesc, Atype, (transa == CUBLAS_OP_N) ? m : k, (transa == CUBLAS_OP_N) ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, (transb == CUBLAS_OP_N) ? k : n, (transb == CUBLAS_OP_N) ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);

    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(const void*));
    check_cuda_error(cublasLtMatmul(
        cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, NULL, 0, stream_));
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
#endif
void cublasMMWrapper::setStream(cudaStream_t stream)
{
    stream_ = stream;
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa,
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
                                         const int         batch_count,
                                         const float       f_alpha,
                                         const float       f_beta)
{
    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    mu_->lock();
    int         is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle_,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                A,
                                                Atype_,
                                                lda,
                                                strideA,
                                                B,
                                                Btype_,
                                                ldb,
                                                strideB,
                                                beta,
                                                C,
                                                Ctype_,
                                                ldc,
                                                strideC,
                                                batch_count,
                                                computeType_,
                                                static_cast<cublasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa,
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
                                         cudaDataType_t    computeType)
{
    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    mu_->lock();
    int         is_fp16_computeType = computeType == CUDA_R_16F ? 1 : 0;
    const void* alpha =
        is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle_,
                                                transa,
                                                transb,
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                A,
                                                AType,
                                                lda,
                                                strideA,
                                                B,
                                                BType,
                                                ldb,
                                                strideB,
                                                beta,
                                                C,
                                                CType,
                                                ldc,
                                                strideC,
                                                batch_count,
                                                computeType,
                                                static_cast<cublasGemmAlgo_t>(info.algoId)));

    mu_->unlock();
}

void cublasMMWrapper::batchedGemm(cublasOperation_t  transa,
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
                                  const int          batch_count)
{
    float f_alpha = static_cast<float>(1.0f);
    float f_beta  = static_cast<float>(0.0f);

    half h_alpha = (half)1.0f;
    half h_beta  = (half)0.0f;

    mu_->lock();
    int         is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    check_cuda_error(cublasGemmBatchedEx(cublas_handle_,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         Atype_,
                                         lda,
                                         B,
                                         Btype_,
                                         ldb,
                                         beta,
                                         C,
                                         Ctype_,
                                         ldc,
                                         batch_count,
                                         computeType_,
                                         static_cast<cublasGemmAlgo_t>(info.algoId)));
    mu_->unlock();
}

void cublasMMWrapper::GemmWithResidualAndBias(cublasOperation_t transa,
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
                                              const void*       bias,
                                              const void*       Residual,
                                              float             f_alpha,
                                              float             f_beta,
                                              const bool        is_gelu,
                                              bool              has_hook)
{
    if (Residual == nullptr) {
        f_beta = 0.0f;
    }
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    mu_->lock();
    // TODO: default cublas libs
    int  is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    bool using_cublasLt      = (Atype_ == CUDA_R_16F) ? true : false;
    int  batch_count         = 1;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    // printf("current gemm m: %d, n: %d, k: %d \n", m, n, k);

    // cublas_algo_map_->insertShape(batch_count, m, n, k);

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t         scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    if (bias != nullptr) {
        // std::cout << "bias != nullptr" << std::endl;
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        if (is_gelu) {
            epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        }
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
        if (is_fp16_computeType) {
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(half*));
        }
        else {
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(float*));
        }
    }
    else if (is_gelu) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
    }
    else if (is_gelu) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
    }

    cublasLtMatmulAlgo_t algo;
    void*                workSpace     = cublas_workspace_;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = 0;
        }
        else {
            cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, info.algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId));

            // cublasLtMatmulHeuristicResult_t heurResult;
            // cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
            //     cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);

            // if (algoStatus != CUBLAS_STATUS_SUCCESS) {
            //     findAlgo = 0;
            // }
        }
    }

    check_cuda_error(cublasLtMatmul(cublaslt_handle_,
                                    operationDesc,
                                    alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    beta,
                                    Residual,
                                    Cdesc,
                                    C,
                                    Cdesc,
                                    (findAlgo == 1 ? (&algo) : NULL),
                                    workSpace,
                                    workspaceSize,
                                    stream_));

    // if (status != CUBLAS_STATUS_SUCCESS) {
    //     status = cublasLtMatmul(cublaslt_handle_,
    //                             operationDesc,
    //                             alpha,
    //                             A,
    //                             Adesc,
    //                             B,
    //                             Bdesc,
    //                             beta,
    //                             Residual,
    //                             Cdesc,
    //                             C,
    //                             Cdesc,
    //                             NULL,
    //                             workSpace,
    //                             workspaceSize,
    //                             stream_);
    // }

    // cudaDeviceSynchronize();

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    sync_check_cuda_error();
    mu_->unlock();

    void * hook_C = this->GemmS3DiffLoraHook(transa, transb, m, n, k, A, lda, B, ldb, C, ldc);
    if(hook_C != nullptr){
        // printf("before add residual\n");
        invokeAddResidual((half*)C, (half*)C, (half*)hook_C, 1, m, n,1,1, stream_);
        // printf("after add residual\n");
    }
}

#ifdef ENABLE_FP8
void cublasMMWrapper::GemmWithResidualAndBias(cublasOperation_t    transa,
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
                                              const void*          bias,
                                              const void*          Residual,
                                              float                f_alpha,
                                              float                f_beta,
                                              const bool           is_gelu)
{
    if (Residual == nullptr) {
        f_beta = 0.0f;
    }
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    mu_->lock();
    // TODO: default cublas libs
    int  is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    bool using_cublasLt      = (Atype_ == CUDA_R_16F) ? true : false;
    int  batch_count         = 1;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    // printf("current gemm m: %d, n: %d, k: %d \n", m, n, k);

    // cublas_algo_map_->insertShape(batch_count, m, n, k);

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(Atype_));

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t         scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(
        &Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(
        &Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));

    if (bias != nullptr) {
        // std::cout << "bias != nullptr" << std::endl;
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        if (is_gelu) {
            epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        }
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
        if (is_fp16_computeType) {
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(half*));
        }
        else {
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(float*));
        }
    }
    else if (is_gelu) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(cublasLtEpilogue_t));
    }

    cublasLtMatmulAlgo_t algo;
    void*                workSpace     = cublas_workspace_;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = 0;
        }
        else {
            cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, info.algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));

            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId));

            // cublasLtMatmulHeuristicResult_t heurResult;
            // cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
            //     cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);

            // if (algoStatus != CUBLAS_STATUS_SUCCESS) {
            //     findAlgo = 0;
            // }
        }
    }

    check_cuda_error(cublasLtMatmul(cublaslt_handle_,
                                    operationDesc,
                                    alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    beta,
                                    Residual,
                                    Cdesc,
                                    C,
                                    Cdesc,
                                    (findAlgo == 1 ? (&algo) : NULL),
                                    workSpace,
                                    workspaceSize,
                                    stream_));

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    sync_check_cuda_error();
    mu_->unlock();
}

#endif  // ENABLE_FP8

bool cublasMMWrapper::isFuseBatchGemm(const int batch_count, const int m, const int k, const int n)
{
    CublasDataType data_type = getCublasDataType(Atype_);

    if (cublas_algo_map_->isExist(batch_count, m, k, n, data_type) == false
        || cublas_algo_map_->isExist(1, m, k, n, data_type) == false) {
        return false;
    }
    else {
        return cublas_algo_map_->getAlgo(batch_count, m, k, n, data_type).exec_time
               < 3 * cublas_algo_map_->getAlgo(1, m, k, n, data_type).exec_time;
    }
}

#ifdef SPARSITY_ENABLED
void cublasMMWrapper::SpGemm(cublasOperation_t transa,
                             cublasOperation_t transb,
                             const int         m,
                             const int         n,
                             const int         k,
                             const void*       A,
                             const void*       B,
                             void*             C)
{
    if (Atype_ != CUDA_R_16F || Btype_ != CUDA_R_16F || Ctype_ != CUDA_R_16F) {
        throw std::runtime_error("\n[FT][ERROR] sparse GEMM only supports FP16 data type now.");
    }
    static bool not_printed_fp32_accumulation_warning = true;
    if (computeType_ != CUDA_R_16F && not_printed_fp32_accumulation_warning) {
        printf("[FT][WARNING] cublasMMWrapper sets to FP32 compute type, "
               "but sparse gemm will use FP16 compute type since cusparselt "
               "supports FP16 accumulation only.\n");
        not_printed_fp32_accumulation_warning = false;
    }
    cusparseOrder_t     order = CUSPARSE_ORDER_COL;
    cusparseOperation_t opA = (transa == CUBLAS_OP_N) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    cusparseOperation_t opB = (transb == CUBLAS_OP_N) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    float    _alpha(1.0f);
    float    _beta(0.0f);

    char mark[256];
    sprintf(mark, "%d_%d_%d_%d", 1, m, n, k);
    if (sp_mat_A_desc_map_.find(mark) != sp_mat_A_desc_map_.end()) {
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &sp_mat_A_desc_map_[mark],
                                                      &sp_mat_B_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      compute_type))
    }
    else {
        // initializing MatDesc takes a lot of time
        cusparseLtMatDescriptor_t matA, matB, matC;
        sp_mat_A_desc_map_[mark] = matA;
        sp_mat_B_desc_map_[mark] = matB;
        sp_mat_C_desc_map_[mark] = matC;
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&cusparselt_handle_,
                                                          &sp_mat_A_desc_map_[mark],
                                                          num_A_rows,
                                                          num_A_cols,
                                                          lda,
                                                          alignment,
                                                          Atype_,
                                                          order,
                                                          CUSPARSELT_SPARSITY_50_PERCENT))
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &sp_mat_B_desc_map_[mark], num_B_rows, num_B_cols, ldb, alignment, Btype_, order))
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &sp_mat_C_desc_map_[mark], num_C_rows, num_C_cols, ldc, alignment, Ctype_, order))
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &sp_mat_A_desc_map_[mark],
                                                      &sp_mat_B_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      compute_type))
    }
    mu_->lock();
    CHECK_CUSPARSE(
        cusparseLtMatmulAlgSelectionInit(&cusparselt_handle_, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = cublas_algo_map_->getSpAlgo(1, num_A_rows, num_B_cols, num_A_cols);
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cusparselt_handle_, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&cusparselt_handle_, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&cusparselt_handle_, &plan, &matmul, &alg_sel, workspace_size))

    void*        d_workspace = nullptr;
    int          num_streams = 1;
    cudaStream_t streams[1]  = {stream_};
    CHECK_CUSPARSE(
        cusparseLtMatmul(&cusparselt_handle_, &plan, &_alpha, A, B, &_beta, C, C, d_workspace, streams, num_streams))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    sync_check_cuda_error();
    mu_->unlock();
}

size_t cublasMMWrapper::getSparseMatrixSize(int m, int k)
{
    // Get a compressed matrix size of shape (m, k) used in cusparselt.
    auto            Atype_     = CUDA_R_16F;
    cusparseOrder_t order      = CUSPARSE_ORDER_COL;
    unsigned        alignment  = 16;
    int             num_A_rows = m;
    int             num_A_cols = k;
    int             lda        = num_A_rows;

    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&cusparselt_handle_,
                                                      &matA,
                                                      num_A_rows,
                                                      num_A_cols,
                                                      lda,
                                                      alignment,
                                                      Atype_,
                                                      order,
                                                      CUSPARSELT_SPARSITY_50_PERCENT));
    size_t compressed_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&cusparselt_handle_, &matA, &compressed_size));
    return compressed_size;
}

void cublasMMWrapper::compressMatrix(const void* input, void* output, const int m, const int k)
{
    cusparseOrder_t           order = CUSPARSE_ORDER_COL;
    cusparseOperation_t       opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseLtMatDescriptor_t matA;
    unsigned                  alignment = 16;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &cusparselt_handle_, &matA, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtSpMMACompress2(&cusparselt_handle_, &matA, true, opA, input, output, stream_))
    sync_check_cuda_error();
}

bool cublasMMWrapper::isUseSparse(const int batch_count, const int m, const int n, const int k)
{
    return cublas_algo_map_->isUseSparse(batch_count, m, n, k);
}
#endif

std::pair<bool, cublasLtMatmulAlgo_t> cublasMMWrapper::findBestAlgo(cublasLtHandle_t       lightHandle,
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
                                                                    cudaStream_t           stream)
{
#if (CUBLAS_VERSION) < 11601
    LYRA_CHECK_WITH_INFO(false, "CUBLAS version too low.");
    return {false, cublasLtMatmulAlgo_t{}};
#else
    size_t  returnSize;
    int32_t pointer_mode;
    cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize);

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t                   preference;
    check_cuda_error(cublasLtMatmulPreferenceCreate(&preference));
    check_cuda_error(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
    uint32_t pointer_mode_mask = 0;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int  return_count = 0;
    auto ret          = cublasLtMatmulAlgoGetHeuristic(lightHandle,
                                              computeDesc,
                                              Adesc,
                                              Bdesc,
                                              Cdesc,
                                              Ddesc,
                                              preference,
                                              heuristics.size(),
                                              heuristics.data(),
                                              &return_count);
    heuristics.resize(return_count);

    // printf("heuristics return_count: %d  \n", return_count);

    std::map<int, std::vector<float>> algo_results;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t              algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        float my_alpha = 1.0f;
        float my_beta  = 0.0f;

        for (int i = 0; i < 11; i++) {
            float duration_ms;
            cudaEventRecord(start_event, stream);
            check_cuda_error(cublasLtMatmul(lightHandle,
                                            computeDesc,
                                            alpha,
                                            A,
                                            Adesc,
                                            B,
                                            Bdesc,
                                            beta,
                                            C,
                                            Cdesc,
                                            D,
                                            Ddesc,
                                            &algo,
                                            cublas_workspace_,
                                            CUBLAS_WORKSPACE_SIZE,
                                            stream));
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[algo_id].push_back(duration_ms);
        }
        std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
    }

    cublasLtMatmulHeuristicResult_t result;
    float                           best_time = INFINITY;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t              algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);
        const auto& results = algo_results[algo_id];

        if (results.size() > 0 && results[5] < best_time) {
            best_time = results[5];
            result    = heuristic;
        }
    }

    printf("searchBestAlgo current best_time: %f ms \n", best_time);

    return {best_time != INFINITY, result.algo};
#endif
}

cublasLtMatmulAlgo_t cublasMMWrapper::searchBestAlgo(cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     const int         m,
                                                     const int         n,
                                                     const int         k,
                                                     const void*       A,
                                                     const int         lda,
                                                     const void*       B,
                                                     const int         ldb,
                                                     void*             C,
                                                     const int         ldc)
{

    // printf("begin searchBestAlgo \n");

    float f_alpha(1.0f);
    float f_beta(0.0f);
    half  h_alpha = (half)(f_alpha);
    half  h_beta  = (half)(f_beta);

    // TODO: default cublas libs
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    int batch_count         = 1;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t         scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    findBestAlgo(cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, stream_);

    cudaDeviceSynchronize();

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    // sync_check_cuda_error();
    return cublasLtMatmulAlgo_t{};
}

cublasMMWrapper::MatrixLayout cublasMMWrapper::createMatrixLayout(cublasLtMatrixLayout_t Mdesc)
{
    size_t       returnSize;
    MatrixLayout m_layout;

    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &std::get<0>(m_layout), sizeof(std::get<0>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &std::get<1>(m_layout), sizeof(std::get<1>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &std::get<2>(m_layout), sizeof(std::get<2>(m_layout)), &returnSize);
    cublasLtMatrixLayoutGetAttribute(
        Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &std::get<3>(m_layout), sizeof(std::get<3>(m_layout)), &returnSize);

    return m_layout;
}

cublasStatus_t cublasMMWrapper::cublasLtMatmulWrapper(cublasLtHandle_t            lightHandle,
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
                                                      cudaStream_t                stream)
{
    cache_idx_t cache_idx{
        computeDesc,
        {createMatrixLayout(Adesc), createMatrixLayout(Bdesc), createMatrixLayout(Cdesc), createMatrixLayout(Ddesc)}};

    cublasLtMatmulAlgo_t algo_value;
    bool                 found_algo = false;
    if (algo == nullptr) {
        if (algo_cache.find(cache_idx) == algo_cache.end()) {
            auto result =
                findBestAlgo(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, stream);
            if (result.first) {
                algo_cache[cache_idx] = result.second;
                algo_value            = result.second;
                found_algo            = true;
            }
        }
        else {
            algo_value = algo_cache[cache_idx];
            found_algo = true;
        }
    }

    return cublasLtMatmul(lightHandle,
                          computeDesc,
                          alpha,
                          A,
                          Adesc,
                          B,
                          Bdesc,
                          beta,
                          C,
                          Cdesc,
                          D,
                          Ddesc,
                          found_algo ? &algo_value : algo,
                          workspace,
                          workspaceSizeInBytes,
                          stream);
}

void cublasMMWrapper::_Int8Gemm(const int     m,
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
                                const bool    per_column_scaling)
{
    /* mode:
     *  - 0: int8 * int8 -> int32 -> int8
     *  - 1: int8 * int8 -> int32 -> int32
     */
#if (CUBLAS_VERSION) < 11601
    LYRA_CHECK_WITH_INFO(false, "CUBLAS version too low.");
#else

    mu_->lock();
    const auto  op_a        = CUBLAS_OP_T;
    const auto  op_b        = CUBLAS_OP_N;
    const auto  dataType    = CUDA_R_8I;
    const auto  resultType  = mode == 0 ? CUDA_R_8I : CUDA_R_32I;
    const auto  computeType = CUBLAS_COMPUTE_32I;
    const auto  scaleType   = mode == 0 ? CUDA_R_32F : CUDA_R_32I;
    const int   batch_count = 1;
    const void* beta;

    int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k, getCublasDataType(dataType));

    cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(batch_count, m, n, k, getCublasDataType(dataType));

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, lda));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, resultType, m, n, ldc));

    check_cuda_error(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));

    auto pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    if (mode == 0) {
        pointer_mode =
            per_column_scaling ? CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST : CUBLASLT_POINTER_MODE_DEVICE;
    }
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &op_b, sizeof(cublasOperation_t)));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

    const int32_t int_one    = 1;
    const int32_t int_zero   = 0;
    const float   float_zero = 0;
    if (mode == 0) {
        beta = per_column_scaling ? &float_zero : NULL;
    }
    else {
        alpha = &int_one;
        beta  = &int_zero;
    }

    cublasLtMatmulAlgo_t algo;
    void*                workSpace     = cublas_workspace_;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    sync_check_cuda_error();
    auto ret = cublasLtMatmulWrapper(cublaslt_handle_,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     NULL,
                                     workSpace,
                                     workspaceSize,
                                     stream_);
    check_cuda_error(ret);
    sync_check_cuda_error();

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    sync_check_cuda_error();
    mu_->unlock();
#endif
}

void cublasMMWrapper::Int8Gemm(const int     m,
                               const int     n,
                               const int     k,
                               const int8_t* A,
                               const int     lda,
                               const int8_t* B,
                               const int     ldb,
                               int8_t*       C,
                               const int     ldc,
                               const float*  alpha,
                               const bool    per_column_scaling)
{
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, alpha, 0, per_column_scaling);
}

void cublasMMWrapper::Int8Gemm(const int     m,
                               const int     n,
                               const int     k,
                               const int8_t* A,
                               const int     lda,
                               const int8_t* B,
                               const int     ldb,
                               int32_t*      C,
                               const int     ldc)
{
    return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, (float*)nullptr, 1, false);
}

int cublasMMWrapper::findBestAlgoFp16(cublasOperation_t        transa,
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
                                      cublasLtMatmulAlgo_info& best_algo)
{
    float f_alpha(1.0f);
    float f_beta(0.0f);
    half  h_alpha = (half)(f_alpha);
    half  h_beta  = (half)(f_beta);

    // TODO: default cublas libs
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    int batch_count         = 1;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    void* workSpace     = cublas_workspace_;
    int   workSpaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent, cudaEventBlockingSync);
    cudaEventCreate(&stopEvent, cudaEventBlockingSync);

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t         scaleType;
    // SplitK value that we are going to try when SplitK is supported for a given
    // algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
// Let try a fixed number of combinations
#define ALGO_COMBINATIONS 50000
    int                AlgoCombinations = ALGO_COMBINATIONS;
    int                AlgoCount        = 0;
    int                kernelRepeats    = 10;  // number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int                nbAlgoIds = 0;
#define ALGO_IDS 100
    int algoIdA[ALGO_IDS];

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    // Request the AlgoId available
    cublasLtMatmulAlgoGetIds(
        cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, ALGO_IDS, algoIdA, &nbAlgoIds);

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(
            cublaslt_handle_, computeType, scaleType, Atype_, Btype_, Ctype_, Ctype_, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int  nbTiles = int(sizeWritten / sizeof(int));
        int* tileA   = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles  = 1;
        }
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int              nbStages = int(sizeWritten / sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages   = 1;
        }
        else {
            cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int) * nbStages, &sizeWritten);
        }
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
            /* Loop over different stages count */
            for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
                /* Loop over the different custom option if any */
                for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                    /* Loop over the CTAs swizzling support */
                    for (int k = 0; k <= swizzlingMax; k++) {
                        int splitK_trial = 0;
                        if (splitkSupport) {
                            splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                        }
                        // Loop over the splitK value over a fixed sequence splitKSequenceA
                        // in addtion to the case where splitK is not enabled
                        for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                            /* Setup attribute of the algo to run */
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                            int splitK_val = 0;
                            int redScheme  = CUBLASLT_REDUCTION_SCHEME_NONE;
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                            if (l > 0) {  // Split-K case
                                splitK_val = splitKSequenceA[l - 1];
                                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                     &splitKSequenceA[l - 1],
                                                                     sizeof(splitKSequenceA[l - 1]));
                                /* Going over all the reduction scheme  */
                                for (redScheme = 1;
                                     redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                     redScheme = redScheme << 1) {
                                    if (redScheme & redMask) {

                                        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                             &redScheme,
                                                                             sizeof(redScheme));

                                        status = perfMatmulRun(cublaslt_handle_,
                                                               operationDesc,
                                                               alpha, /* host or device pointer */
                                                               A,
                                                               Adesc,
                                                               B,
                                                               Bdesc,
                                                               beta, /* host or device pointer */
                                                               C,
                                                               Cdesc,
                                                               C,
                                                               Cdesc,
                                                               algo,
                                                               kernelRepeats,
                                                               workSpace,
                                                               workSpaceSize,
                                                               perfResults[AlgoCount],
                                                               stream_,
                                                               startEvent,
                                                               stopEvent);

                                        perfResults[AlgoCount].status = status;
                                        if (status == CUBLAS_STATUS_SUCCESS)
                                            AlgoCount++;
                                    }  // end if
                                }  // end for
                            }
                            else {  // Non-splitK case
                                /* if user preference is ok with workspace */
                                if (AlgoCount < AlgoCombinations) {

                                    status = perfMatmulRun(cublaslt_handle_,
                                                           operationDesc,
                                                           alpha, /* host or device pointer */
                                                           A,
                                                           Adesc,
                                                           B,
                                                           Bdesc,
                                                           beta, /* host or device pointer */
                                                           C,
                                                           Cdesc,
                                                           C,
                                                           Cdesc,
                                                           algo,
                                                           kernelRepeats,
                                                           workSpace,
                                                           workSpaceSize,
                                                           perfResults[AlgoCount],
                                                           stream_,
                                                           startEvent,
                                                           stopEvent);

                                    perfResults[AlgoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS)
                                        AlgoCount++;
                                }
                            }
                        }  // end l
                    }  // end k
                }  // end customOption
            }
        }  // end tileIdx
        delete[] tileA;
    }  // end idx
    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0; i < AlgoCount; i++) {
        // memset(tmp_cstr, 0, sizeof tmp_cstr);
        // sprintf(tmp_cstr, "FP16 gemm IO cublasLt %03d : ", i);
        // tmp_str = string(tmp_cstr);
        // gemm_test_result += tmp_str;
        printPerfStructure(m, n, k, perfResults[i], best_algo, i);
        break;
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc)
        cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc)
        cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc)
        cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc)
        cublasLtMatmulDescDestroy(operationDesc);
    if (startEvent)
        cudaEventDestroy(startEvent);
    if (stopEvent)
        cudaEventDestroy(stopEvent);

    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

// template<typename T>
// void cublasMMWrapper::profileGemmInt8(const int maxM, const int n, const int k)
// {
//     if (std::is_same<T, half>::value) {
//         fp16_int8_matmul_runner_->configurePlugin(maxM, k, n);
//         fp16_int8_matmul_runner_->initialize();
//         cudaDeviceSynchronize();
//     }
//     else if (typeid(T) == typeid(__nv_bfloat16)) {
//         bf16_int8_matmul_runner_->configurePlugin(maxM, k, n);
//         bf16_int8_matmul_runner_->initialize();
//         cudaDeviceSynchronize();
//     }
// }

// template<typename T>
// void cublasMMWrapper::runGemmInt8(int8_t const* A,
//                                   int8_t const* B,
//                                   float const*  alphaCol,
//                                   float const*  alphaRow,
//                                   void*         C,
//                                   int           m,
//                                   int           n,
//                                   int           k,
//                                   char*         workspacePtr,
//                                   cudaStream_t  stream)
// {
//     if (std::is_same<T, half>::value) {
//         fp16_int8_matmul_runner_->RunGemm(A, B, alphaCol, alphaRow, C, m, n, k, workspacePtr, stream);
//     }
//     else if (typeid(T) == typeid(__nv_bfloat16)) {
//         bf16_int8_matmul_runner_->RunGemm(A, B, alphaCol, alphaRow, C, m, n, k, workspacePtr, stream);
//     }
// }

}  // namespace lyradiff
