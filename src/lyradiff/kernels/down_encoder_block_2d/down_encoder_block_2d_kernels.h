#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeDownSamplePad(T*           dst,
                         const T*     input,
                         const size_t batch_size,
                         const size_t height,
                         const size_t width,
                         const size_t nchannels,
                         cudaStream_t stream);

}  // namespace lyradiff