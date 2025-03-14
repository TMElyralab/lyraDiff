#include "src/lyradiff/utils/quantization.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace lyradiff {

template<typename T>
void invokeSqInputQuant(int8_t*      dst,
                        const T*     src,
                        const float* pre_quant_scale,
                        const float* input_quant_scale,
                        QuantMode    quant_mode,
                        size_t       batch_size,
                        size_t       seq_len,
                        size_t       dim,
                        cudaStream_t stream);

template<typename T>
void invokeSqOutputDequant(T*             dst,
                           const int32_t* src,
                           const float*   input_quant_scale,
                           const float*   weight_quant_scale,
                           QuantMode      quant_mode,
                           size_t         batch_size,
                           size_t         seq_len,
                           size_t         dim,
                           cudaStream_t   stream);

template<typename T>
void invokeGetINT8WeightScale(float*       new_weight_scale,
                              int8_t*      quantized_weight,
                              const T*     weight,
                              const float* pre_quant_scale,
                              const size_t d_out,
                              const size_t d_in,
                              cudaStream_t stream = 0);
}  // namespace lyradiff