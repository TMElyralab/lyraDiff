


#pragma once

namespace cutlass
{

enum class WeightOnlyQuantOp
{
    UNDEFINED,
    PER_COLUMN_SCALE_ONLY,
    FINEGRAINED_SCALE_ONLY,
    FINEGRAINED_SCALE_AND_ZEROS
};

constexpr bool isFinegrained(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS || op == WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
}

constexpr bool hasZero(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
}

} // namespace cutlass
