#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {
template<typename T>
void fused_linear_geglu(T*           output,
                          const T*     input,
                          const T*     weight1,
                          const T*     bias1,
                          const T*     weight2,
                          const T*     bias2,
                          size_t       input_b,
                          size_t       input_seqlen,
                          size_t       input_dim,
                          size_t       output_dim,
                          void*        cublas_workspace_,
                          const bool   allow_half_precision,
                          cudaStream_t stream);

}  // namespace lyradiff
