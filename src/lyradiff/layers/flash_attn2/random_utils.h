#pragma once

#include "random_state.h"
#include <cstdint>
#include <tuple>

// In-kernel call to retrieve philox seed and offset from a PhiloxCudaState instance whether
// that instance was created with graph capture underway or not.
// See Note [CUDA Graph-safe RNG states].
//
// The raw definition lives in its own file so jit codegen can easily copy it.
#if defined(__CUDA_ACC__) or defined(__CUDA_ARCH__)
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace at {
namespace cuda {
namespace philox {

inline DEVICE std::tuple<uint64_t, uint64_t> unpack(PhiloxCudaState arg)
{
    if (arg.captured_) {
        // static_cast avoids "warning: invalid narrowing conversion from "long" to "unsigned long".
        // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire kernel.
        // For most threads' reads it will hit in cache, so it shouldn't hurt performance.
        return std::make_tuple(arg.seed_, static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
    }
    else {
        return std::make_tuple(arg.seed_, arg.offset_.val);
    }
}

}  // namespace philox
}  // namespace cuda
}  // namespace at