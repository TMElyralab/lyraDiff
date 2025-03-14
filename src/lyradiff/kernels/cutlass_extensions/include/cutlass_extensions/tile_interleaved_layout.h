

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/pitch_linear_coord.h"

namespace cutlass
{
namespace layout
{

template <int RowsPerTile, int ColumnsInterleaved>
struct ColumnMajorTileInterleave
{
    static constexpr int kRowsPerTile = RowsPerTile;
    static constexpr int kColumnsInterleaved = ColumnsInterleaved;
};

template <class T>
struct IsColumnMajorTileInterleave
{
    static constexpr bool value = false;
};

template <int U, int V>
struct IsColumnMajorTileInterleave<ColumnMajorTileInterleave<U, V>>
{
    static constexpr bool value = true;
};

} // namespace layout
} // namespace cutlass
