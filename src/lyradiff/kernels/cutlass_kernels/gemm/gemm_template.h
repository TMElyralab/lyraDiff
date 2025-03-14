

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

// clang-format off
#include <cutlass/gemm/device/gemm.h>
// clang-format on

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "src/lyradiff/kernels/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "src/lyradiff/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "src/lyradiff/kernels/cutlass_kernels/gemm/gemm.h"
// #include "src/lyrallms/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/cuda_utils.h"

#include <chrono>
#include <sstream>

namespace tk  = lyradiff;
namespace tkc = lyradiff::cutlass_extensions;

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages>
void genericGemmKernelLauncher(void const*            A,
                               void const*            B,
                               void*                  C,
                               void const*            bias,
                               int                    m,
                               int                    n,
                               int                    k,
                               tkc::CutlassGemmConfig gemmConfig,
                               char*                  workspace,
                               size_t                 workspaceBytes,
                               cudaStream_t           stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementAccumulator     = float;               // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA          = cutlass::half_t;     // <- data type of elements in input matrix A
    using ElementInputB          = cutlass::half_t;     // <- data type of elements in input matrix B
    using ElementOutput          = cutlass::half_t;     // <- data type of elements in output matrix D

    // Note that if the output is column major, the bias has to be per row. i.e. every row has different bias.
    // If the output is row major, the bias has to be per column, i.e. every column has different bias.
    // Below list some other notices:
    //
    // Note this example only works for ColumnMajor output because
    //   1) we only have row major epilogue.
    //   2) we swap A and B if the output is column major then we can still use the
    //      row major epilogue.
    //   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
    //   4) we can use the existing OutputIterator to load 1xM bias vector.

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = arch;

    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock = ThreadblockShape;
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = WarpShape;
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

    // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
    //
    //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
    //

    bool hasBias = bias != nullptr;

    constexpr auto kScaleType = cutlass::epilogue::thread::ScaleType::Default;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                    128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                                    ElementAccumulator,
                                                                    ElementAccumulator,
                                                                    kScaleType>;

    const ElementAccumulator alpha = ElementAccumulator(1);
    const ElementAccumulator beta  = ElementAccumulator(hasBias ? 1 : 0);

    // Number of pipelines you want to use
    constexpr int NumStages = Stages;

    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                             LayoutInputA,
                                             ElementInputB,
                                             LayoutInputB,
                                             ElementOutput,
                                             LayoutOutput,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOp,
                                             SwizzleThreadBlock,
                                             NumStages,
                                             128 / cutlass::sizeof_bits<ElementOutput>::value,
                                             128 / cutlass::sizeof_bits<ElementOutput>::value,
                                             true>;  // supports k split

    typename cutlass::TensorRef<const ElementInputA, cutlass::layout::RowMajor> tensor_a(
        reinterpret_cast<const ElementInputA*>(A), cutlass::layout::RowMajor(k));
    typename cutlass::TensorRef<const ElementInputA, cutlass::layout::ColumnMajor> tensor_b(
        reinterpret_cast<const ElementInputB*>(B), cutlass::layout::ColumnMajor(k));

    typename cutlass::TensorRef<const ElementOutput, cutlass::layout::RowMajor> tensor_bias(
        reinterpret_cast<const ElementOutput*>(bias), cutlass::layout::RowMajor(0));

    typename cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> nullptr_ref{};

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_out(reinterpret_cast<ElementOutput*>(C),
                                                                            cutlass::layout::RowMajor(n));

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    typename Gemm::Arguments arguments{
        problem_size,  // <- problem size of matrix multiplication
        tensor_a,      // <- reference to matrix A on device
        tensor_b,      // <- reference to matrix B on device

        tensor_bias,  // <- the C matrix is treated as the bias vector. We can enable the GEMM
                      //    to project away the N dimension by setting the stride to zero.

        tensor_out,                  // <- reference to matrix D on device
        {alpha, beta},               // <- alpha
        gemmConfig.split_k_factor};  // <- k-dimension split factor

    Gemm gemm;
    // // TODO: handle that
    // if (gemm.get_workspace_size(args) > workspaceBytes) {
    //     LYRA_LOG_WARNING(
    //         "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
    //     // If requested split-k factor will require more workspace bytes, revert to standard gemm.
    //     args.batch_count = 1;
    // }

    auto can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string errMsg =
            "gemm cutlass kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[lyradiff Error][gemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(arguments, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess) {
        std::string errMsg =
            "Failed to initialize cutlass gemm. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[lyradiff Error][gemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess) {
        std::string errMsg =
            "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[lyradiff Error][gemm Runner] " + errMsg);
    }
}

template<typename T,
         typename arch,
         typename ThreadblockShape,
         typename WarpShape,
         int  Stages,
         typename Enable = void>
struct dispatchStages {
    static void dispatch(void const*            A,
                         void const*            B,
                         void*                  C,
                         void const*            bias,
                         int                    m,
                         int                    n,
                         int                    k,
                         tkc::CutlassGemmConfig gemmConfig,
                         char*                  workspace,
                         size_t                 workspaceBytes,
                         cudaStream_t           stream)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string errMsg = "Cutlass gemm. Not instantiates for arch " + std::to_string(arch::kMinComputeCapability)
                             + " with stages set to " + std::to_string(Stages);
        throw std::runtime_error("[lyradiff Error][dispatchStages::dispatch] " + errMsg);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
struct dispatchStages<T, arch, ThreadblockShape, WarpShape, 2> {
    static void dispatch(void const*            A,
                         void const*            B,
                         void*                  C,
                         void const*            bias,
                         int                    m,
                         int                    n,
                         int                    k,
                         tkc::CutlassGemmConfig gemmConfig,
                         char*                  workspace,
                         size_t                 workspaceBytes,
                         cudaStream_t           stream)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        genericGemmKernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(
            A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
    }
};

template<typename T, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatchStages<T,
                      cutlass::arch::Sm80,
                      ThreadblockShape,
                      WarpShape,
                      Stages,
                      typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(void const*            A,
                         void const*            B,
                         void*                  C,
                         void const*            bias,
                         int                    m,
                         int                    n,
                         int                    k,
                         tkc::CutlassGemmConfig gemmConfig,
                         char*                  workspace,
                         size_t                 workspaceBytes,
                         cudaStream_t           stream)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        genericGemmKernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(
            A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void dispatchGemmConfig(void const*            A,
                        void const*            B,
                        void*                  C,
                        void const*            bias,
                        int                    m,
                        int                    n,
                        int                    k,
                        tkc::CutlassGemmConfig gemmConfig,
                        char*                  workspace,
                        size_t                 workspaceBytes,
                        cudaStream_t           stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.stages) {
        case 2:
            using DispatcherStages2 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case 3:
            using DispatcherStages3 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case 4:
            using DispatcherStages4 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case 5:
            using DispatcherStages5 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 5>;
            DispatcherStages5::dispatch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case 6:
            using DispatcherStages6 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 6>;
            DispatcherStages6::dispatch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        default:
            std::string errMsg = "dispatchGemmConfig does not support stages " + std::to_string(gemmConfig.stages);
            throw std::runtime_error("[lyradiff Error][dispatch_gemm_config] " + errMsg);
            break;
    }
}

template<typename T, typename arch>
void dispatchGemmToCutlass(void const*            A,
                           void const*            B,
                           void*                  C,
                           void const*            bias,
                           int                    m,
                           int                    n,
                           int                    k,
                           tkc::CutlassGemmConfig gemmConfig,
                           char*                  workspace,
                           size_t                 workspaceBytes,
                           cudaStream_t           stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemmConfig.tile_config) {
        case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<32, 64, 64>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x32:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 32>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        // case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x32:
        //     dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 32>>(
        //         A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
        //     break;
        case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x64x64:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x64x32:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 64, 32>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        // case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x32:
        //     dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 32, 32>>(
        //         A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
        //     break;
        case tkc::CutlassTileConfig::CtaShape128x64x32_WarpShape64x64x32:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 64, 32>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::CtaShape128x64x32_WarpShape64x32x32:
            dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>>(
                A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
            break;
        case tkc::CutlassTileConfig::Undefined:
            throw std::runtime_error("[lyradiff Error] [dispatch_gemm_to_cutlass] gemm config undefined.");
            break;
        case tkc::CutlassTileConfig::ChooseWithHeuristic:
            throw std::runtime_error(
                "[lyradiff Error][dispatch_gemm_to_cutlass] gemm config should have already been set by "
                "heuristic.");
            break;
        default:
            throw std::runtime_error(
                "[lyradiff Error][dispatch_gemm_to_cutlass] Config is invalid for int8 GEMM.");
            break;
    }
}

template<typename T>
CutlassGemmRunner<T>::CutlassGemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    tk::check_cuda_error(cudaGetDevice(&device));
    mSm = tk::getSMVersion();
    tk::check_cuda_error(cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
}

template<typename T>
CutlassGemmRunner<T>::~CutlassGemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void CutlassGemmRunner<T>::dispatchToArch(void const*            A,
                                          void const*            B,
                                          void*                  C,
                                          void const*            bias,
                                          int                    m,
                                          int                    n,
                                          int                    k,
                                          tkc::CutlassGemmConfig gemmConfig,
                                          char*                  workspace,
                                          size_t                 workspaceBytes,
                                          cudaStream_t           stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (mSm >= 75 && mSm < 80) {
        dispatchGemmToCutlass<T, cutlass::arch::Sm75>(
            A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
    }
    else if (mSm >= 80 && mSm <= 90) {
        dispatchGemmToCutlass<T, cutlass::arch::Sm80>(
            A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
    }
    else {
        throw std::runtime_error(
            "[lyradiff Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}

template<typename T>
void CutlassGemmRunner<T>::gemm(void const*            A,
                                void const*            B,
                                void*                  C,
                                void const*            bias,
                                int                    m,
                                int                    n,
                                int                    k,
                                tkc::CutlassGemmConfig gemmConfig,
                                char*                  workspace,
                                size_t                 workspaceBytes,
                                cudaStream_t           stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatchToArch(A, B, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream);
}

template<typename T>
std::vector<tkc::CutlassGemmConfig> CutlassGemmRunner<T>::getConfigs() const
{
    static constexpr bool               isWeightOnly = false;
    std::vector<tkc::CutlassGemmConfig> candidateConfigs =
        get_candidate_configs(mSm, isWeightOnly, mSm <= 70, false, SPLIT_K_LIMIT);
    return candidateConfigs;
}

template<typename T>
size_t CutlassGemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    int const maxGridM = cutlass::ceil_div(m, MIN_M_TILE);
    int const maxGridN = cutlass::ceil_div(m, MIN_N_TILE);
    // We need 4 bytes per block in the worst case. We launch SPLIT_K_LIMIT in z dim.
    return static_cast<size_t>(maxGridM * maxGridN * SPLIT_K_LIMIT * 4);
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
