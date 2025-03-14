#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {
void invokeFusedQuantizeAndLoraDownSimple(float*       Cptr,
                                          uint32_t*    quantized_input,
                                          float*       quantized_scale,
                                          const void*  Aptr,
                                          const void*  Bptr,
                                          const float* Smoothptr,
                                          int          m,
                                          int          n,
                                          int          k,
                                          int          group_dim = 64,
                                          cudaStream_t stream    = 0);

void invokeFusedW4A4GemmAndLoraUp(void*        Dptr,
                                  const void*  Aptr,
                                  const void*  Bptr,
                                  const void*  LoraAptr,
                                  const void*  LoraBptr,
                                  const float* AweightScale,
                                  const float* BweightScale,
                                  int          m,
                                  int          n,
                                  int          k,
                                  int          lora_k,
                                  int          groups,
                                  cudaStream_t stream);

}  // namespace lyradiff
