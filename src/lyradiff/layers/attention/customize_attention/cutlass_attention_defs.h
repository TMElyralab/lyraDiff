#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/contrib/args_pack_def.h"
#include "cutlass/contrib/epilogue/threadblock/softmax_epilogue.h"
#include "cutlass/contrib/gemm/device/gemm_grouped.h"
#include "cutlass/contrib/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"
#include <vector>

#include "cutlass/util/device_dump.h"
#include "cutlass_attention_operators.h"
#include "src/lyradiff/common.h"

namespace lyradiff {
namespace cutlass_ops {

template<int SeqLen, int SizePerHead, typename ArchTag>
struct DefaultAttentionConfig {
    static constexpr int kSeqLen = SeqLen;
    static_assert(!(kSeqLen % 64), "Cutlass Attention requires seqlen % 64 == 0");
    static constexpr int kSizePerHead = SizePerHead;
    static constexpr int kStages0     = 3;
    static constexpr int kStages1     = 4;

    using InstructionShape  = cutlass::gemm::GemmShape<16, 8, 8>;
    using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 128, 16>;
    using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
    using WarpShape0        = cutlass::gemm::GemmShape<64, 64, 16>;
    using WarpShape1        = cutlass::gemm::GemmShape<32, 32, 32>;
    using TileShapeDef0     = cutlass::contrib::TileShapeDef<ThreadblockShape0, WarpShape0, InstructionShape>;
    using TileShapeDef1     = cutlass::contrib::TileShapeDef<ThreadblockShape1, WarpShape1, InstructionShape>;
};

/// Attention Core Class
template<int SeqLen, int SizePerHead, typename ArchTag, int AttenMaskAlignment>
struct CutlassAttentionCore {
    static const int kSeqLen      = SeqLen;
    static const int kSizePerHead = SizePerHead;

    using Config = DefaultAttentionConfig<kSeqLen, kSizePerHead, ArchTag>;

    using Element               = cutlass::half_t;
    using ElementSoftmaxCompute = float;

    using EpilogueOp = typename cutlass::epilogue::thread::
        LinearCombination<Element, 128 / cutlass::sizeof_bits<Element>::value, Element, Element>;

    static const int kAlignment = 8;
    using SoftmaxEpilogueOp     = SoftmaxPartialEpilogueOp<Element, kAlignment, Element, ElementSoftmaxCompute>;

    template<int Alignment>
    using TensorDefRowMajor = cutlass::contrib::
        TensorDef<Element, cutlass::layout::RowMajor, Alignment, AttentionTensorParamGeneratorOp<Element>>;
    template<int Alignment>
    using TensorDefColMajor = cutlass::contrib::
        TensorDef<Element, cutlass::layout::ColumnMajor, Alignment, AttentionTensorParamGeneratorOp<Element>>;

    using IdentityPrologueDef = cutlass::contrib::PrologueDef<>;

    using SoftmaxPrologueDef = cutlass::contrib::PrologueDef<SoftmaxPartialPrologueOp<Element, ElementSoftmaxCompute>>;
    using SoftmaxEpilogueDef = cutlass::contrib::EpilogueDef<TensorDefRowMajor<kAlignment>, SoftmaxEpilogueOp>;

    using GemmKernel0 = typename cutlass::contrib::gemm::kernel::DefaultGemmGrouped<
        /// Tensor defs
        TensorDefRowMajor<kAlignment>,  // A
        TensorDefColMajor<kAlignment>,  // B
        TensorDefRowMajor<kAlignment>,  // C
        TensorDefRowMajor<kAlignment>,  // D
        /// Prologue defs
        IdentityPrologueDef,  // A
        IdentityPrologueDef,  // B
        /// Epilogue defs
        SoftmaxEpilogueDef,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        typename Config::TileShapeDef0,
        Config::kStages0,
        cutlass::arch::OpMultiplyAdd,
        AttentionGemmProblemSize<SizePerHead, 1, /*IsGemm0=*/true>,
        AttentionBatchCountGeneratorOp>::GemmGroupedKernel;
    using GemmKernel1 = typename cutlass::contrib::gemm::kernel::DefaultGemmGrouped<
        /// Tensor defs
        TensorDefRowMajor<kAlignment>,  // A
        TensorDefRowMajor<kAlignment>,  // B
        TensorDefRowMajor<kAlignment>,  // C
        TensorDefRowMajor<kAlignment>,  // D
        /// Prologue defs
        SoftmaxPrologueDef,   // A
        IdentityPrologueDef,  // B
        /// Epilogue defs
        cutlass::contrib::EpilogueDef<TensorDefRowMajor<kAlignment>, EpilogueOp>,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        typename Config::TileShapeDef1,
        Config::kStages1,
        cutlass::arch::OpMultiplyAdd,
        AttentionGemmProblemSize<SizePerHead, kAlignment, /*IsGemm0=*/false>,
        AttentionBatchCountGeneratorOp>::GemmGroupedKernel;
    using Gemm0 = typename cutlass::contrib::gemm::device::GemmGrouped<GemmKernel0>;
    using Gemm1 = typename cutlass::contrib::gemm::device::GemmGrouped<GemmKernel1>;
};

}  // namespace cutlass_ops
}  // namespace lyradiff