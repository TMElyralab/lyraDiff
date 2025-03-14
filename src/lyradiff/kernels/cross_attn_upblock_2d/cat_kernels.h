#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

namespace lyradiff {

template<typename T>
void invokeCatByChannel(T*           dst,
                        const T*     src1,
                        const T*     src2,
                        const size_t channel1,
                        const size_t channel2,
                        const size_t height,
                        const size_t width,
                        const int    batch_size,
                        cudaStream_t stream);

}  // namespace lyradiff
