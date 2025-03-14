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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile this
    iterator visits maybe partial, then the remaining tiles are complete. So, we
    only need to compute the predicates twice, once before the first tile and
    once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template<typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
         typename Element_,    ///< Element data type
         bool UseCUDAStore = false>
class PredicatedTileIterator {
public:
    using ThreadMap = ThreadMap_;
    using Shape     = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout         = layout::RowMajor;
    using TensorRef      = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index       = typename Layout::Index;
    using LongIndex   = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads           = ThreadMap::kThreads;
    static int const kIterations        = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element,
                           ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup
                               * ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    /// Uses a non-template class
    struct Params: cutlass::epilogue::threadblock::PredicatedTileIteratorParams {
        using Base = cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout):
            Base(layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
                 cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>())
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Base const& base): Base(base) {}
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask()
        {
            enable();
        }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// A thread's starting row position (assuming steady-state predicates have been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates have been computed)
    Index thread_start_col_;

    /// Internal state counter
    int state_[3];

    //
    // Static asserts about internal strides
    //

    static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
    static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
    static_assert(sizeof(Params::stride) == 8, "Expected 64b strides");

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    PredicatedTileIterator(Params const& params,
                           Element*      pointer,
                           TensorCoord   extent,
                           int           thread_idx,
                           TensorCoord   threadblock_offset = TensorCoord()):
        params_(params)
    {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_row_       = extent.row();
        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] = ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
        }

        // Null pointer performs no accesses
        if (!pointer) {
            mask_.clear();
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) + LongIndex(thread_offset.row()) * LongIndex(params_.stride)
                        + LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_HOST_DEVICE
    cutlass::MatrixCoord thread_offset() const
    {
        return {thread_start_row_, thread_start_col_};
    }

    CUTLASS_HOST_DEVICE
    Params const& params() const
    {
        return params_;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset)
    {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const
    {
        uint8_t*    byte_pointer = byte_pointer_;
        AccessType* frag_ptr     = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup
                                     + cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

                    AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];

                        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }
    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) const
    {
        load_with_byte_offset(frag, 0);
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const
    {
        uint8_t*          byte_pointer = byte_pointer_;
        AccessType const* frag_ptr     = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup
                                     + cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

                    AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];

                        if (UseCUDAStore) {
                            if (guard) {
                                memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess] =
                                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
                            }
                        }
                        else {
                            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                                (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                                guard);
                        }
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) const
    {
        store_with_byte_offset(frag, 0);
    }

    /// Need to get the thread start row from the tile iterator
    CUTLASS_DEVICE
    int32_t thread_start_row() const
    {
        return thread_start_row_;
    }

    /// Extent of the matrix in rows
    CUTLASS_DEVICE
    Index extent_row() const
    {
        return extent_row_;
    }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++()
    {
        ++state_[0];
        byte_pointer_ += params_.advance_row;
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow
                                     * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask()
    {
        mask_.enable();
    }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) const
    {
        mask = mask_;
    }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        mask_ = mask;
    }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | InterleavedPredicatedTileIterator | ForwardTileIterator
///
template<typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
         typename Element_,    ///< Element data type
         int InterleavedN      ///< Number of Interleaved N
         >
class InterleavedPredicatedTileIterator {
public:
    using ThreadMap = ThreadMap_;

    using Element = Element_;

    using Layout         = layout::ColumnMajorInterleaved<InterleavedN>;
    using TensorRef      = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index       = typename Layout::Index;
    using LongIndex   = typename Layout::LongIndex;
    using TensorCoord = layout::PitchLinearCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads           = ThreadMap::kThreads;
    static int const kIterations        = ThreadMap::Iterations::kCount;

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    /// Uses a non-template class
    struct Params: cutlass::epilogue::threadblock::InterleavedPredicatedTileIteratorParams {
        using Base = InterleavedPredicatedTileIteratorParams;

        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout):
            Base(layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
                 cutlass::epilogue::threadblock::make_InterleavedPredicatedTileIteratorDesc<Element, ThreadMap>())
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Base const& base): Base(base) {}
    };

    /// Mask object
    struct Mask {
        static int const kCount = (ThreadMap::Iterations::kContiguous < 8) ? 8 : ThreadMap::Iterations::kContiguous;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask()
        {
            enable();
        }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in columns
    Index extent_col_;

    /// A thread's starting column position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_col_;

    /// Internal iteration counter
    int iteration_contiguous_;

    int iteration_strided_;

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    InterleavedPredicatedTileIterator(
        Params const& params, Element* pointer, TensorCoord extent, int thread_idx, TensorCoord threadblock_offset):
        params_(params)
    {
        TensorCoord thread_offset =
            ThreadMap::initial_offset(thread_idx)
            + TensorCoord(threadblock_offset.contiguous() * InterleavedN, threadblock_offset.strided() / InterleavedN);

        extent_col_       = extent.strided() / InterleavedN;
        thread_start_col_ = thread_offset.strided();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            mask_.predicates[c] = ((thread_offset.contiguous() + ThreadMap::Delta::kContiguous * c)
                                   < (extent.contiguous() * InterleavedN));
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer)
                        + LongIndex(thread_offset.strided()) * LongIndex(params_.stride)
                        + LongIndex(thread_offset.contiguous()) * sizeof(AccessType) / kElementsPerAccess;

        // Initialize internal state counter
        iteration_contiguous_ = iteration_strided_ = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset)
    {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag)
    {
        uint8_t*    byte_pointer   = byte_pointer_;
        AccessType* frag_ptr       = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer);

        int col_offset = iteration_strided_ * ThreadMap::Delta::kStrided;

        bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);

        bool guard = col_guard && mask_.predicates[iteration_contiguous_];

        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(*frag_ptr, (void*)memory_pointer, guard);
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag)
    {
        uint8_t*          byte_pointer   = byte_pointer_;
        AccessType const* frag_ptr       = reinterpret_cast<AccessType const*>(&frag);
        AccessType*       memory_pointer = reinterpret_cast<AccessType*>(byte_pointer);

        int col_offset = iteration_strided_ * ThreadMap::Delta::kStrided;

        bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);

        bool guard = col_guard && mask_.predicates[iteration_contiguous_];

        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(*frag_ptr, (void*)memory_pointer, guard);
    }

    /// Overrides the internal iteration index
    CUTLASS_HOST_DEVICE
    void set_iteration_index(int iteration)
    {
        iteration_contiguous_ = iteration % ThreadMap::Iterations::kContiguous;
        iteration_strided_    = iteration / ThreadMap::Iterations::kContiguous;
    }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    InterleavedPredicatedTileIterator& operator++()
    {
        ++iteration_contiguous_;
        byte_pointer_ += params_.advance_row;

        if (iteration_contiguous_ == ThreadMap::Iterations::kContiguous) {
            iteration_contiguous_ = 0;
            ++iteration_strided_;
            byte_pointer_ += params_.advance_column;

            if (iteration_strided_ == ThreadMap::Iterations::kStrided) {
                iteration_strided_ = 0;
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask()
    {
        mask_.enable();
    }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        mask = mask_;
    }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        mask_ = mask;
    }
};

///////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | InterleavedMaskedTileIterator | ForwardTileIterator
///
template<typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
         typename Element_,    ///< Element data type
         int InterleavedN      ///< Number of Interleaved N
         >
class InterleavedConvPredicatedTileIterator {
public:
    using ThreadMap = ThreadMap_;

    using Element = Element_;

    using Layout         = layout::TensorNCxHWx<InterleavedN>;
    using TensorRef      = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index       = typename Layout::Index;
    using LongIndex   = typename Layout::LongIndex;
    using TensorCoord = Tensor4DCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads           = ThreadMap::kThreads;
    static int const kIterations        = ThreadMap::Iterations::kCount;

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride_col;  ///< stride in bytes between columns
        LongIndex stride_row;  ///< stride in bytes between rows

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(typename Layout::Stride stride_)
        {
            stride_col = stride_[1];
            stride_row = stride_[2];

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params()
        {
            initialize(cutlass::make_Coord(0, 0, 0));
        }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
        {
            initialize(layout.stride());
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = (ThreadMap::Iterations::kRow < 8) ? 8 : ThreadMap::Iterations::kRow;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask()
        {
            enable();
        }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in columns
    Index extent_col_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// Extent of the matrix tile in pq
    Index extent_pq_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_col_;

    /// Internal iteration counter
    LongIndex iteration_row_;
    LongIndex iteration_col_;

    uint32_t pq_mul_;

    uint32_t pq_shr_;

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    InterleavedConvPredicatedTileIterator(
        Params const& params, Element* pointer, TensorCoord extent, int thread_idx, MatrixCoord threadblock_offset):
        params_(params)
    {
        MatrixCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_col_ = extent.c();
        extent_pq_  = extent.h() * extent.w();
        extent_row_ = extent.n() * extent_pq_;

        find_divisor(pq_mul_, pq_shr_, extent_pq_);

        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < ThreadMap::Iterations::kRow; ++r) {
            mask_.predicates[r] = ((thread_offset.row() + ThreadMap::Delta::kRow * r) < extent_row_);
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer)
                        + ((thread_start_col_ / InterleavedN) * params_.stride_col + (thread_start_col_ % InterleavedN))
                              * sizeof_bits<Element>::value / 8;

        // Initialize internal state counter
        iteration_row_ = iteration_col_ = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset)
    {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag)
    {
        int  col_offset = iteration_col_ * ThreadMap::Delta::kColumn;
        bool col_guard  = ((thread_start_col_ + col_offset) < extent_col_);
        bool guard      = col_guard && mask_.predicates[iteration_row_];

        int n, pq_rem;

        fast_divmod(
            n, pq_rem, thread_start_row_ + iteration_row_ * ThreadMap::Delta::kRow, extent_pq_, pq_mul_, pq_shr_);

        uint8_t* byte_pointer =
            byte_pointer_ + (n * params_.stride_row + pq_rem * InterleavedN) * sizeof_bits<Element>::value / 8;
        AccessType*       frag_ptr       = reinterpret_cast<AccessType*>(&frag);
        AccessType const* memory_pointer = reinterpret_cast<AccessType const*>(byte_pointer);

        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(*frag_ptr, (void*)memory_pointer, guard);
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag)
    {
        int  col_offset = iteration_col_ * ThreadMap::Delta::kColumn;
        bool col_guard  = ((thread_start_col_ + col_offset) < extent_col_);
        bool guard      = col_guard && mask_.predicates[iteration_row_];

        int n, pq_rem;

        fast_divmod(
            n, pq_rem, thread_start_row_ + iteration_row_ * ThreadMap::Delta::kRow, extent_pq_, pq_mul_, pq_shr_);

        uint8_t* byte_pointer =
            byte_pointer_ + (n * params_.stride_row + pq_rem * InterleavedN) * sizeof_bits<Element>::value / 8;
        AccessType const* frag_ptr       = reinterpret_cast<AccessType const*>(&frag);
        AccessType*       memory_pointer = reinterpret_cast<AccessType*>(byte_pointer);

        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(*frag_ptr, (void*)memory_pointer, guard);
    }

    /// Overrides the internal iteration index
    CUTLASS_HOST_DEVICE
    void set_iteration_index(int iteration)
    {
        iteration_row_ = iteration % ThreadMap::Iterations::kRow;
        iteration_col_ = iteration / ThreadMap::Iterations::kRow;
    }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    InterleavedConvPredicatedTileIterator& operator++()
    {
        ++iteration_row_;

        if (iteration_row_ == ThreadMap::Iterations::kRow) {
            iteration_row_ = 0;
            ++iteration_col_;
            byte_pointer_ += params_.stride_col;

            if (iteration_col_ == ThreadMap::Iterations::kColumn) {
                iteration_col_ = 0;
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask()
    {
        mask_.enable();
    }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        mask = mask_;
    }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        mask_ = mask;
    }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace contrib
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
