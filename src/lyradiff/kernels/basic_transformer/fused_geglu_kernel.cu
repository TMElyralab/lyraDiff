#include <cutlass/cutlass.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/linear_combination_with_elementwise.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include "fused_geglu_kernel.h"

#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace cutlass {
namespace epilogue {
namespace thread {

template<typename ElementOutput_,
         int Count,
         template<typename>
         typename Activation,
         typename ElementAccumulator_ = ElementOutput_,
         typename ElementCompute_     = ElementOutput_,
         FloatRoundStyle Round        = FloatRoundStyle::round_to_nearest>
class RightActivationAndMul {
public:
    using ElementOutput      = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute     = ElementCompute_;

    static int const kCount   = Count;
    using FragmentOutput      = Array<ElementOutput, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using ComputeFragment     = Array<ElementCompute, kCount>;

    static FloatRoundStyle const kRound = Round;

    struct Params {};

private:
    ElementCompute alpha_;
    ElementCompute beta_;

public:
    CUTLASS_HOST_DEVICE
    RightActivationAndMul(Params const& /*params*/) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const
    {
        return true;
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count)
    {
        assert(false);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const
    {
        NumericArrayConverter<ElementOutput, ElementAccumulator, kCount, Round> accumulator_to_output;

        FragmentOutput converted_lhs = accumulator_to_output(lhs);
        FragmentOutput converted_rhs = accumulator_to_output(rhs);

        Activation<FragmentOutput>          act;
        cutlass::multiplies<FragmentOutput> mul;
        auto                                act_rhs = act(converted_rhs);
        return mul(act_rhs, converted_lhs);
    }

    CUTLASS_HOST_DEVICE
    ElementOutput operator()(ElementAccumulator const& lhs, ElementAccumulator const& rhs) const
    {
        ElementOutput                      convert_lhs(lhs);
        ElementOutput                      convert_rhs(rhs);
        Activation<ElementOutput>          act;
        cutlass::multiplies<ElementOutput> mul;
        auto                               act_rhs = act(convert_rhs);
        return mul(act_rhs, convert_lhs);
    }
};
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

namespace lyradiff {

template<typename T>
struct GetCutlassType {
    using type = T;
};

template<>
struct GetCutlassType<half> {
    using type = cutlass::half_t;
};

template<typename Acc, typename Arch, template<typename> typename Activation>
void DualGemmGegluHalf(cudaStream_t stream,
                       int32_t      m,
                       int32_t      n,
                       int32_t      k,
                       const void*  x,
                       const void*  w,
                       const void*  v,
                       const void*  b,
                       const void*  c,
                       void*        y,
                       void*        cublas_workspace_)
{
    constexpr int  kStages       = 5;
    constexpr bool kSplitKSerial = false;
    constexpr bool kUseBias      = true;
    using ElementOperandA        = cutlass::half_t;
    using ElementOperandB        = cutlass::half_t;
    using ElementOutput          = cutlass::half_t;
    using ElementAccumulator     = Acc;
    using ElementCompute         = Acc;
    using ThreadblockShape       = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape              = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape       = cutlass::gemm::GemmShape<16, 8, 16>;

    constexpr auto kScaleType = kUseBias ? cutlass::epilogue::thread::ScaleType::NoBetaScaling :
                                           (
                                               // No bias
                                               kSplitKSerial ? cutlass::epilogue::thread::ScaleType::Default :
                                                               cutlass::epilogue::thread::ScaleType::Nothing);
    using EpilogueOutputOp0 =
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementAccumulator,
                                                     ElementCompute,
                                                     kScaleType>;
    using EpilogueOutputOp1 =
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementAccumulator,
                                                     ElementCompute,
                                                     kScaleType>;
    using EpilogueOutputOp2 =
        cutlass::epilogue::thread::RightActivationAndMul<ElementOutput,
                                                         128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                         Activation,
                                                         ElementOutput,
                                                         ElementCompute>;

    const ElementCompute alpha0 = ElementCompute(1);
    const ElementCompute beta0  = ElementCompute(kUseBias ? 1 : 0);
    const ElementCompute alpha1 = ElementCompute(1);
    const ElementCompute beta1  = ElementCompute(kUseBias ? 1 : 0);

    // Optionally, we might not need intermediate GEMM outputs
    constexpr bool kStoreD0 = false;
    constexpr bool kStoreD1 = false;
    using DualGemm          = cutlass::gemm::device::DualGemm<ElementOperandA,
                                                     cutlass::layout::RowMajor,
                                                     ElementOperandB,
                                                     cutlass::layout::ColumnMajor,
                                                     cutlass::layout::ColumnMajor,
                                                     ElementOutput,
                                                     cutlass::layout::RowMajor,
                                                     ElementAccumulator,
                                                     cutlass::arch::OpClassTensorOp,
                                                     Arch,
                                                     ThreadblockShape,
                                                     WarpShape,
                                                     InstructionShape,
                                                     EpilogueOutputOp0,
                                                     EpilogueOutputOp1,
                                                     EpilogueOutputOp2,
                                                     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
                                                     kStages,
                                                     kStoreD0,
                                                     kStoreD1,
                                                     kSplitKSerial>;

    int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;

    typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_a0(
        reinterpret_cast<const cutlass::half_t*>(x), cutlass::layout::RowMajor(k));
    typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b0(
        reinterpret_cast<const cutlass::half_t*>(w), cutlass::layout::ColumnMajor(k));
    typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::ColumnMajor> tensor_b1(
        reinterpret_cast<const cutlass::half_t*>(v), cutlass::layout::ColumnMajor(k));
    typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias0(
        reinterpret_cast<const cutlass::half_t*>(b), cutlass::layout::RowMajor(0));
    typename cutlass::TensorRef<const ElementOperandA, cutlass::layout::RowMajor> tensor_bias1(
        reinterpret_cast<const cutlass::half_t*>(c), cutlass::layout::RowMajor(0));

    typename cutlass::TensorRef<ElementOperandA, cutlass::layout::RowMajor> nullptr_ref{};

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_out(reinterpret_cast<cutlass::half_t*>(y),
                                                                            cutlass::layout::RowMajor(n));

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    typename DualGemm::Arguments arguments{cutlass::gemm::DualGemmMode::kGemm,
                                           problem_size,
                                           tensor_a0,
                                           tensor_b0,
                                           tensor_bias0,
                                           nullptr_ref,
                                           tensor_b1,
                                           tensor_bias1,
                                           nullptr_ref,
                                           tensor_out,
                                           {alpha0, beta0},
                                           {alpha1, beta1},
                                           {},
                                           split_k_slices};

    DualGemm dual_gemm_op;
    dual_gemm_op.initialize(arguments, cublas_workspace_, stream);
    dual_gemm_op(stream);
}

template<typename Acc, typename Arch>
bool TryDispatchDualGemmImplActivation(cudaStream_t       stream,
                                       const std::string& activation,
                                       int32_t            m,
                                       int32_t            n,
                                       int32_t            k,
                                       const void*        x,
                                       const void*        w,
                                       const void*        v,
                                       const void*        b,
                                       const void*        c,
                                       void*              y,
                                       void*              cublas_workspace_)
{
    if (activation == "fast_gelu") {
        DualGemmGegluHalf<Acc, Arch, cutlass::epilogue::thread::GELU_taylor>(
            stream, m, n, k, x, w, v, b, c, y, cublas_workspace_);
        return true;
    }
    else if (activation == "gelu") {
        DualGemmGegluHalf<Acc, Arch, cutlass::epilogue::thread::GELU>(
            stream, m, n, k, x, w, v, b, c, y, cublas_workspace_);
        return true;
    }
    else {
        return false;
    }
}

template<typename T, typename Arch>
bool TryDispatchDualGemmImplAccType(cudaStream_t       stream,
                                    const std::string& activation,
                                    int32_t            m,
                                    int32_t            n,
                                    int32_t            k,
                                    const T*           x,
                                    const T*           w,
                                    const T*           v,
                                    const T*           b,
                                    const T*           c,
                                    T*                 y,
                                    void*              cublas_workspace_,
                                    const bool         allow_half_precision)
{
    if (std::is_same<T, half>::value) {
        if (allow_half_precision) {
            return TryDispatchDualGemmImplActivation<cutlass::half_t, Arch>(
                stream, activation, m, n, k, x, w, v, b, c, y, cublas_workspace_);
        }
        else {
            return TryDispatchDualGemmImplActivation<float, Arch>(
                stream, activation, m, n, k, x, w, v, b, c, y, cublas_workspace_);
        }
    }
    else {
        return false;
    }
}

template<typename T, typename Arch>
bool TryDispatchDualGemmImpl(cudaStream_t       stream,
                             const std::string& activation,
                             int32_t            m,
                             int32_t            n,
                             int32_t            k,
                             const T*           x,
                             const T*           w,
                             const T*           v,
                             const T*           b,
                             const T*           c,
                             T*                 y,
                             void*              cublas_workspace_,
                             const bool         allow_half_precision)
{
    if (m % 8 == 0 && n % 8 == 0 && k % 8 == 0 && reinterpret_cast<uintptr_t>(x) % (8 * sizeof(T)) == 0
        && reinterpret_cast<uintptr_t>(w) % (8 * sizeof(T)) == 0
        && reinterpret_cast<uintptr_t>(v) % (8 * sizeof(T)) == 0
        && reinterpret_cast<uintptr_t>(b) % (8 * sizeof(T)) == 0
        && reinterpret_cast<uintptr_t>(c) % (8 * sizeof(T)) == 0
        && reinterpret_cast<uintptr_t>(y) % (8 * sizeof(T)) == 0) {
        return TryDispatchDualGemmImplAccType<T, Arch>(
            stream, activation, m, n, k, x, w, v, b, c, y, cublas_workspace_, allow_half_precision);
    }
    else {
        return false;
    }
}

template<typename T>
void fused_linear_geglu(T*           output,
                        const T*     input,
                        const T*     weight1,
                        const T*     bias1,
                        const T*     weight2,
                        const T*     bias2,
                        size_t       input_b,
                        size_t       input_seqlen,
                        size_t       input_dim,
                        size_t       output_dim,
                        void*        cublas_workspace_,
                        const bool   allow_half_precision,
                        cudaStream_t stream)
{

    const size_t m = input_b * input_seqlen;
    const size_t n = output_dim;
    const size_t k = input_dim;

    TryDispatchDualGemmImpl<T, cutlass::arch ::Sm80>(stream,
                                                     "fast_gelu",
                                                     m,
                                                     n,
                                                     k,
                                                     input,
                                                     weight1,
                                                     weight2,
                                                     bias1,
                                                     bias2,
                                                     output,
                                                     cublas_workspace_,
                                                     allow_half_precision);
}

#define INSTANTIATE_FUSED_LINEAR_GEGLU(T)                                                                            \
    template void fused_linear_geglu(T*           output,                                                              \
                                     const T*     input,                                                               \
                                     const T*     weight1,                                                             \
                                     const T*     bias1,                                                               \
                                     const T*     weight2,                                                             \
                                     const T*     bias2,                                                               \
                                     size_t       input_b,                                                             \
                                     size_t       input_seqlen,                                                        \
                                     size_t       input_dim,                                                           \
                                     size_t       output_dim,                                                          \
                                     void*        cublas_workspace_,                                                   \
                                     const bool   allow_half_precision,                                                \
                                     cudaStream_t stream)

INSTANTIATE_FUSED_LINEAR_GEGLU(float);
INSTANTIATE_FUSED_LINEAR_GEGLU(half);
#undef INSTANTIATE_FUSED_LINEAR_GEGLU

}  // namespace lyradiff
