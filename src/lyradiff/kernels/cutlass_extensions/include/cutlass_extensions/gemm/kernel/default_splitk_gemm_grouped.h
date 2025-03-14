



#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"

#include "cutlass/layout/permute.h"

#include "splitk_gemm_grouped.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace kernel
{

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_ = GroupScheduleMode::kDeviceOnly,
    /// Operation performed by GEMM
    typename Operator = typename device::DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA_, ElementB_,
        ElementC_, ElementAccumulator>::Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    ///
    typename Enable = void>
struct DefaultSplitkGemmGrouped;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Permute result D
    typename PermuteDLayout>
struct DefaultSplitkGemmGrouped<ElementA, LayoutA,
    ComplexTransform::kNone, // transform A
    kAlignmentA, ElementB, LayoutB,
    ComplexTransform::kNone, // transform B
    kAlignmentB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
    InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, GroupScheduleMode_, Operator, SharedMemoryClear,
    PermuteDLayout, typename platform::enable_if<!cutlass::is_complex<ElementAccumulator>::value>::type>
{

    // If true, we must construct a 'transposed-and-exchanged' Mma operator.
    static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;

    using MapArguments = kernel::detail::MapArguments<ElementA, LayoutA, ComplexTransform::kNone, kAlignmentA, ElementB,
        LayoutB, ComplexTransform::kNone, kAlignmentB, LayoutC, kInternalTranspose>;

    // Define the default GEMM kernel
    using DefaultGemmKernel = typename kernel::DefaultGemm<typename MapArguments::ElementA,
        typename MapArguments::LayoutA, MapArguments::kAlignmentA, typename MapArguments::ElementB,
        typename MapArguments::LayoutB, MapArguments::kAlignmentB, ElementC, typename MapArguments::LayoutC,
        ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, true, Operator, SharedMemoryClear, false, 
        false,                                                                
        false,                                                                
        PermuteDLayout>::GemmKernel;

    /// Define the kernel in terms of the default kernel
    using GemmKernel = kernel::SplitkGemmGrouped<typename DefaultGemmKernel::Mma, typename DefaultGemmKernel::Epilogue,
        ThreadblockSwizzle, GroupScheduleMode_, kInternalTranspose>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
