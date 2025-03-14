


#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass_extensions/arch/mma.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"

namespace cutlass
{
namespace gemm
{
namespace warp
{

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements,
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    arch::OpMultiplyAddDequantizeInterleavedBToA, PartitionsK, AccumulatorsInRowMajor>
{

private:
    // Shape for computing the FP16s
    using ComputeInstructionShape = InstructionShape_;

    // Chosen so we get K=16 for int8 and K=32 for int4.
    static constexpr int LoadInstructionK = 8 * sizeof_bits<ElementA>::value / sizeof_bits<ElementB>::value;

    // Shape for loading the narrow data type from shared memory
    using LoadInstructionShape = GemmShape<InstructionShape_::kM, InstructionShape_::kN, LoadInstructionK>;

public:
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
        cutlass::arch::Mma<InstructionShape_, 32, ElementA, cutlass::layout::RowMajor, ElementA,
            cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
        cutlass::MatrixShape<1, 1>>;

    // Define the warp-level tensor op
    using Type = cutlass::gemm::warp::MmaTensorOpComputeBWithF16<WarpShape_, ElementA, LayoutA, ElementB, LayoutB,
        ElementC, LayoutC, Policy, LoadInstructionShape, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
