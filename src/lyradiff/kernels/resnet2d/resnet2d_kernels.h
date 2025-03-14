#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T>
void invokeAddConvAndTemb(T*           dst,
                          const T*     conv_out,
                          const T*     temb_out,
                          const T*     temb_bias,
                          const size_t batch_size,
                          const size_t height,
                          const size_t width,
                          const size_t nchannels,
                          cudaStream_t stream);

template<typename T>
void invokeAddTwoConvOutScale(T*           dst,
                              const T*     conv1_out,
                              const T*     conv2_out,
                              const size_t batch_size,
                              const size_t height,
                              const size_t width,
                              const size_t nchannels,
                              const float  scale,
                              cudaStream_t stream);
}  // namespace lyradiff