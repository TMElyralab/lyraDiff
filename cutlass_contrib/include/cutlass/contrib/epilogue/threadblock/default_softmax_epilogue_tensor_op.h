/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"

// #include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"

#include "../warp/softmax_fragment_iterator_tensor_op.h"
#include "../warp/softmax_fragment_iterator_volta_tensor_op.h"
#include "default_thread_map_tensor_op_for_attr_mask.h"
#include "output_tile_thread_map_for_attr_mask.h"
#include "softmax_epilogue.h"
#include "softmax_epilogue_shm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template<typename Shape_, typename WarpMmaTensorOp_, int PartitionsK, typename OutputOp_, int ElementsPerAccess>
struct DefaultSoftmaxEpilogueTensorOp {
    using Shape                         = Shape_;
    using WarpMmaTensorOp               = WarpMmaTensorOp_;
    static int const kPartitionsK       = PartitionsK;
    using OutputOp                      = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    using ElementOutput                 = typename OutputOp::ElementOutput;
    using LayoutC                       = typename WarpMmaTensorOp::LayoutC;
    using ElementAccumulator            = typename WarpMmaTensorOp::ElementC;

    static_assert(!(Shape::kN % WarpMmaTensorOp::Shape::kN), "Threadblock kN must be divisible by Warp kN");
    static int constexpr kPartitionsN   = Shape::kN / WarpMmaTensorOp::Shape::kN;
    static bool constexpr kUseShmReduce = kPartitionsN != 1;
    //
    // Thread map
    //

    using OutputTileThreadMap =
        typename threadblock::DefaultThreadMapTensorOpForAttrMask<Shape,
                                                                  typename WarpMmaTensorOp::Shape,
                                                                  kPartitionsK,
                                                                  ElementOutput,
                                                                  kElementsPerAccess>::Type;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIterator<OutputTileThreadMap, ElementOutput>;

    using AccumulatorFragmentIterator =
        epilogue::warp::SoftmaxFragmentIteratorTensorOp<typename WarpMmaTensorOp::Shape,
                                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                                        LayoutC>;

    //
    // Define the epilogue
    //
    using Epilogue = typename std::conditional<kUseShmReduce,
                                               threadblock::SoftmaxEpilogueShm<Shape,
                                                                               typename WarpMmaTensorOp::Shape,
                                                                               OutputTileIterator,
                                                                               AccumulatorFragmentIterator,
                                                                               OutputOp>,
                                               threadblock::SoftmaxEpilogue<Shape,
                                                                            WarpMmaTensorOp,
                                                                            kPartitionsK,
                                                                            OutputTileIterator,
                                                                            AccumulatorFragmentIterator,
                                                                            OutputOp>>::type;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace contrib
}  // namespace cutlass
