/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is copied from NVIDIA/cutlass and modified.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
*/
#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace gemm {
namespace kernel {

template<typename ProblemSizeOperator_, typename BatchCountOperator_>
struct GemmGroupedProblemVisitor {
    using ProblemSizeOperator = ProblemSizeOperator_;
    using BatchCountOperator  = BatchCountOperator_;

    struct Params {
        typename ProblemSizeOperator::Params problem_size_op;
        typename BatchCountOperator::Params  batch_count_op;
        int32_t                              problem_count;

        //
        // Methods
        //

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params(): problem_count(0) {}

        /// Ctor
        CUTLASS_HOST_DEVICE
        Params(typename ProblemSizeOperator::Params problem_size_op,
               typename BatchCountOperator::Params  batch_count_op,
               int32_t                              problem_count):
            problem_size_op(problem_size_op), batch_count_op(batch_count_op), problem_count(problem_count)
        {
        }
    };

    struct SharedStorage {
        //
        // Nothing for now. As an optimization step, we could consider parallel
        // argmin or prefix sums across the block.
        //
    };

    //
    // Data members
    //

    SharedStorage&      shared_storage;
    Params const&       params;
    ProblemSizeOperator problem_size_op;
    BatchCountOperator  batch_count_op;

    cutlass::MatrixCoord threadblock_shape;

    int32_t tile_idx;
    int32_t group_problem_start;
    // tile range of current group of problems: [group_tile_start, group_tile_end)
    int32_t group_tile_start;
    int32_t group_tile_end;
    int32_t presum_in_group;
    int32_t count_in_group;
    int32_t problem_idx;
    // tile range of current problem: [problem_tile_start, problem_tile_end)
    int32_t problem_tile_start;
    int32_t problem_tile_end;

    //
    // Methods
    //
    CUTLASS_DEVICE
    GemmGroupedProblemVisitor(Params const&        params_,
                              SharedStorage&       shared_storage_,
                              cutlass::MatrixCoord threadblock_shape_,
                              int32_t              block_idx):
        shared_storage(shared_storage_),
        params(params_),
        problem_size_op(params.problem_size_op),
        batch_count_op(params.batch_count_op),
        threadblock_shape(threadblock_shape_),
        tile_idx(block_idx),
        group_problem_start(-32),
        group_tile_start(0),
        group_tile_end(0),
        problem_idx(-1),
        problem_tile_start(0),
        problem_tile_end(0)
    {
    }

    /// Get the grid shape
    CUTLASS_HOST_DEVICE
    static cutlass::gemm::GemmCoord
    grid_shape(cutlass::gemm::GemmCoord problem, cutlass::MatrixCoord const& block_shape, int32_t batch_count)
    {
        return cutlass::gemm::GemmCoord(((problem.m() - 1 + block_shape.row()) / block_shape.row()),
                                        ((problem.n() - 1 + block_shape.column()) / block_shape.column()),
                                        batch_count);
    }

    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord grid_shape(cutlass::gemm::GemmCoord problem)
    {
        return grid_shape(problem, threadblock_shape, batch_count_op(problem_idx));
    }

    /// Returns true if there is a tile to compute
    CUTLASS_DEVICE
    bool next_tile()
    {
        if (tile_idx < problem_tile_end) {
            // tile idx still in current problem range
            return true;
        }

        int lane_idx = threadIdx.x % 32;

        while (group_tile_end <= tile_idx) {
            group_problem_start += 32;
            if (group_problem_start > params.problem_count) {
                return false;
            }
            group_tile_start = group_tile_end;
            // int32_t prev_sum = 0;
            int32_t problem_idx = group_problem_start + lane_idx;
            count_in_group      = 0;
            if (problem_idx < params.problem_count) {
                cutlass::gemm::GemmCoord problem = problem_size_op(problem_idx);
                cutlass::gemm::GemmCoord grid    = grid_shape(problem, threadblock_shape, batch_count_op(problem_idx));
                count_in_group                   = grid.m() * grid.n() * grid.k();
            }
            presum_in_group = count_in_group;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < 32; i <<= 1) {
                int32_t val = __shfl_up_sync(0xffffffff, presum_in_group, i);
                if (lane_idx >= i) {
                    presum_in_group += val;
                }
            }
            int32_t total = __shfl_sync(0xffffffff, presum_in_group, 31);
            group_tile_end += total;
        }

        int32_t problem_idx_in_group =
            __popc(__ballot_sync(0xffffffff, presum_in_group <= tile_idx - group_tile_start));
        problem_idx        = group_problem_start + problem_idx_in_group;
        int32_t count      = __shfl_sync(0xffffffff, count_in_group, problem_idx_in_group);
        int32_t presum     = __shfl_sync(0xffffffff, presum_in_group, problem_idx_in_group);
        problem_tile_start = group_tile_start + presum - count;
        problem_tile_end   = group_tile_start + presum;
        return true;
    }

    /// Gets the global tile index
    CUTLASS_HOST_DEVICE
    int64_t tile_index() const
    {
        return tile_idx;
    }

    /// Gets the index of the problem
    CUTLASS_HOST_DEVICE
    int32_t problem_index() const
    {
        return problem_idx;
    }

    /// Returns the problem size for the current problem
    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord problem_size() const
    {
        cutlass::gemm::GemmCoord problem = problem_size_op(problem_idx);
        return problem;
    }

    CUTLASS_HOST_DEVICE
    int64_t threadblock_index() const
    {
        return tile_idx - problem_tile_start;
    }

    CUTLASS_DEVICE
    void advance(int32_t grid_size)
    {
        tile_idx += grid_size;
    }
};

template<typename ProblemSizeOperator_,
         typename BatchCountOperator_,
         typename ParamOperatorA_,
         typename ParamOperatorB_,
         typename ParamOperatorC_,
         typename ParamOperatorD_>
struct GemmParamsDef {
    using ProblemSizeOperator = ProblemSizeOperator_;
    using BatchCountOperator  = BatchCountOperator_;
    using ParamOperatorA      = ParamOperatorA_;
    using ParamOperatorB      = ParamOperatorB_;
    using ParamOperatorC      = ParamOperatorC_;
    using ParamOperatorD      = ParamOperatorD_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Mma_,       ///! Threadblock-scoped matrix multiply-accumulate
         typename Epilogue_,  ///! Epilogue
         typename ParamsDef_>
struct GemmGrouped {
public:
    using Mma              = Mma_;
    using Epilogue         = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ParamsDef        = ParamsDef_;

    using ElementA     = typename Mma::IteratorA::Element;
    using LayoutA      = typename Mma::IteratorA::Layout;
    using PrologueDefA = typename Mma::PrologueDefA;

    using ElementB     = typename Mma::IteratorB::Element;
    using LayoutB      = typename Mma::IteratorB::Layout;
    using PrologueDefB = typename Mma::PrologueDefB;

    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC  = typename Epilogue::OutputTileIterator::Layout;
    using ElementD = ElementC;
    using LayoutD  = LayoutC;

    using ProblemVisitor =
        GemmGroupedProblemVisitor<typename ParamsDef::ProblemSizeOperator, typename ParamsDef::BatchCountOperator>;

    // Type definitions about the mainloop.
    using Operator         = typename Mma::Operator;
    using OperatorClass    = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape        = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag          = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;

    /// Warp count (concept: GemmShape)
    using WarpCount               = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    //
    // Structures
    //

    /// Argument structure
    struct Arguments {
        //
        // Data members
        //

        cutlass::gemm::GemmUniversalMode mode;
        int                              problem_count;
        int                              threadblock_count;

        typename ParamsDef::ProblemSizeOperator::Params problem_size_op;
        typename ParamsDef::BatchCountOperator::Params  batch_count_op;

        typename PrologueDefA::Operator::Params prologue_op_A;
        typename PrologueDefB::Operator::Params prologue_op_B;
        typename EpilogueOutputOp::Params       output_op;

        typename ParamsDef::ParamOperatorA::Params param_A_op;
        typename ParamsDef::ParamOperatorB::Params param_B_op;
        typename ParamsDef::ParamOperatorC::Params param_C_op;
        typename ParamsDef::ParamOperatorD::Params param_D_op;

        //
        // Methods
        //

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments(): mode(cutlass::gemm::GemmUniversalMode::kGemm), problem_count(0), threadblock_count(0) {}

        /// Ctor
        CUTLASS_HOST_DEVICE
        Arguments(cutlass::gemm::GemmUniversalMode                mode,
                  int                                             problem_count,
                  int                                             threadblock_count,
                  typename ParamsDef::ProblemSizeOperator::Params problem_size_op,
                  typename ParamsDef::BatchCountOperator::Params  batch_count_op,
                  typename PrologueDefA::Operator::Params         prologue_op_A,
                  typename PrologueDefB::Operator::Params         prologue_op_B,
                  typename EpilogueOutputOp::Params               output_op,
                  typename ParamsDef::ParamOperatorA::Params      param_A_op,
                  typename ParamsDef::ParamOperatorB::Params      param_B_op,
                  typename ParamsDef::ParamOperatorC::Params      param_C_op,
                  typename ParamsDef::ParamOperatorD::Params      param_D_op):
            mode(mode),
            problem_count(problem_count),
            threadblock_count(threadblock_count),
            problem_size_op(problem_size_op),
            batch_count_op(batch_count_op),
            prologue_op_A(prologue_op_A),
            prologue_op_B(prologue_op_B),
            output_op(output_op),
            param_A_op(param_A_op),
            param_B_op(param_B_op),
            param_C_op(param_C_op),
            param_D_op(param_D_op)
        {
        }
    };

    //
    // Structure for precomputing values in host memory and passing to kernels
    //

    /// Parameters structure
    struct Params {
        typename ProblemVisitor::Params  problem_visitor;
        cutlass::gemm::GemmUniversalMode mode;
        int                              threadblock_count;

        typename PrologueDefA::Operator::Params prologue_op_A;
        typename PrologueDefB::Operator::Params prologue_op_B;
        typename EpilogueOutputOp::Params       output_op;

        typename ParamsDef::ParamOperatorA::Params param_A_op;
        typename ParamsDef::ParamOperatorB::Params param_B_op;
        typename ParamsDef::ParamOperatorC::Params param_C_op;
        typename ParamsDef::ParamOperatorD::Params param_D_op;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params(): mode(cutlass::gemm::GemmUniversalMode::kGemm) {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args, void* workspace = nullptr):
            problem_visitor(args.problem_size_op, args.batch_count_op, args.problem_count),
            mode(args.mode),
            threadblock_count(args.threadblock_count),
            prologue_op_A(args.prologue_op_A),
            prologue_op_B(args.prologue_op_B),
            output_op(args.output_op),
            param_A_op(args.param_A_op),
            param_B_op(args.param_B_op),
            param_C_op(args.param_C_op),
            param_D_op(args.param_D_op)
        {
        }

        CUTLASS_HOST_DEVICE
        void update(Arguments const& args, void* workspace = nullptr)
        {
            problem_visitor =
                typename ProblemVisitor::Params(args.problem_size_op, args.batch_count_op, args.problem_count);
            threadblock_count = args.threadblock_count;
            prologue_op_A     = args.prologue_op_A;
            prologue_op_B     = args.prologue_op_B;
            output_op         = args.output_op;
            param_A_op        = args.params_A_op;
            param_B_op        = args.params_B_op;
            param_C_op        = args.params_C_op;
            param_D_op        = args.params_D_op;
        }
    };

    /// Shared memory storage structure
    union SharedStorage {
        typename ProblemVisitor::SharedStorage problem_visitor;
        typename Mma::SharedStorage            main_loop;
        typename Epilogue::SharedStorage       epilogue;
    };

public:
    //
    // Methods
    //

    CUTLASS_DEVICE
    GemmGrouped() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(cutlass::gemm::GemmCoord const& problem_size)
    {
        return Status::kSuccess;
    }

    static Status can_implement(Arguments const& args)
    {
        return Status::kSuccess;
    }

    static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
    {
        return 0;
    }

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        //
        // Problem visitor.
        //
        ProblemVisitor problem_visitor(
            params.problem_visitor, shared_storage.problem_visitor, {Mma::Shape::kM, Mma::Shape::kN}, blockIdx.x);

        // Outer 'persistent' loop to iterate over tiles
        while (problem_visitor.next_tile()) {
            cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
            int32_t                  problem_idx  = problem_visitor.problem_index();
            int32_t                  cta_idx      = int32_t(problem_visitor.threadblock_index());

            cutlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

            int batch_idx = cta_idx / (grid_shape.m() * grid_shape.n());

            cta_idx %= grid_shape.m() * grid_shape.n();
            cutlass::gemm::GemmCoord threadblock_offset(
                int(cta_idx / grid_shape.n()) * Mma::Shape::kM, int(cta_idx % grid_shape.n()) * Mma::Shape::kN, 0);

            typename ParamsDef::ParamOperatorA param_A_op(params.param_A_op);
            typename ParamsDef::ParamOperatorB param_B_op(params.param_B_op);
            typename ParamsDef::ParamOperatorC param_C_op(params.param_C_op);
            typename ParamsDef::ParamOperatorD param_D_op(params.param_D_op);

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{
                threadblock_offset.m(),
                0,
            };

            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            // Compute position within threadblock
            int thread_idx = threadIdx.x;

            cutlass::contrib::TensorParams<ElementA> params_A = param_A_op(problem_idx, batch_idx);
            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(
                LayoutA(params_A.ldm), params_A.ptr, {problem_size.m(), problem_size.k()}, thread_idx, tb_offset_A);

            cutlass::contrib::TensorParams<ElementB> params_B = param_B_op(problem_idx, batch_idx);
            typename Mma::IteratorB                  iterator_B(
                LayoutB(params_B.ldm), params_B.ptr, {problem_size.k(), problem_size.n()}, thread_idx, tb_offset_B);

            typename Mma::FragmentC accumulators;

            accumulators.clear();

            // Broadcast the warp_id computed by lane 0 to ensure dependent code
            // is compiled as warp-uniform.
            int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

            int lane_idx = threadIdx.x % 32;

            int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);

            int warp_idx_m = warp_idx_mn % WarpCount::kM;
            int warp_idx_n = warp_idx_mn / WarpCount::kM;

            cutlass::gemm::GemmCoord warp_offset =
                threadblock_offset
                + cutlass::gemm::GemmCoord(warp_idx_m * WarpShape::kM, warp_idx_n * WarpShape::kN, 0);

            //
            // Matrix multiply phase
            //

            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

            // Compute threadblock-scoped matrix multiply-add
            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            // Wait for all threads to finish their epilogue phases from the previous tile.
            __syncthreads();

            if (gemm_k_iterations > 0) {
                typename PrologueDefA::Operator prologue_op_A(params.prologue_op_A,
                                                              problem_idx,
                                                              batch_idx,
                                                              {problem_size.m(), problem_size.k()},
                                                              {warp_offset.m(), 0});
                typename PrologueDefB::Operator prologue_op_B(params.prologue_op_B,
                                                              problem_idx,
                                                              batch_idx,
                                                              {problem_size.k(), problem_size.n()},
                                                              {0, warp_offset.n()});

                // Compute threadblock-scoped matrix multiply-add
                mma(gemm_k_iterations,
                    accumulators,
                    iterator_A,
                    iterator_B,
                    prologue_op_A,
                    prologue_op_B,
                    accumulators);
            }

            //
            // Epilogue
            //

            EpilogueOutputOp output_op = [&]() -> auto
            {
                if constexpr (std::is_constructible<EpilogueOutputOp,
                                                    typename EpilogueOutputOp::Params const&,
                                                    int,
                                                    int>::value) {
                    return EpilogueOutputOp(params.output_op, problem_idx, batch_idx);
                }
                else {
                    return EpilogueOutputOp(params.output_op);
                }
            }
            ();

            cutlass::contrib::TensorParams<ElementC> params_C = param_C_op(problem_idx, batch_idx);
            // Tile iterator loading from source tensor.
            typename Epilogue::OutputTileIterator iterator_C(
                typename Epilogue::OutputTileIterator::Params(LayoutC(params_C.ldm)),
                params_C.ptr,
                problem_size.mn(),
                thread_idx,
                threadblock_offset.mn());

            cutlass::contrib::TensorParams<ElementD> params_D = param_D_op(problem_idx, batch_idx);
            // Tile iterator writing to destination tensor.
            typename Epilogue::OutputTileIterator iterator_D(
                typename Epilogue::OutputTileIterator::Params(LayoutD(params_D.ldm)),
                params_D.ptr,
                problem_size.mn(),
                thread_idx,
                threadblock_offset.mn());

            Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

            // Execute the epilogue operator to update the destination tensor.
            epilogue(output_op, iterator_D, accumulators, iterator_C);

            // Next tile
            problem_visitor.advance(gridDim.x);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace contrib
}  // namespace cutlass
