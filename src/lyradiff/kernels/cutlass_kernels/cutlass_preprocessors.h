

#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

enum class QuantType {
    NONE,
    INT8_WEIGHT_ONLY,
    PACKED_INT4_WEIGHT_ONLY
};
int get_bits_in_quant_type(QuantType quant_type);

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void permute_B_rows_for_mixed_gemm(int8_t*                    permuted_quantized_tensor,
                                   int8_t const*              quantized_tensor,
                                   std::vector<size_t> const& shape,
                                   QuantType                  quant_type,
                                   const int64_t              arch_version);

void subbyte_transpose(int8_t*                    transposed_quantized_tensor,
                       int8_t const*              quantized_tensor,
                       std::vector<size_t> const& shape,
                       QuantType                  quant_type);

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void preprocess_weights_for_mixed_gemm(int8_t*                    preprocessed_quantized_weight,
                                       int8_t const*              row_major_quantized_weight,
                                       std::vector<size_t> const& shape,
                                       QuantType                  quant_type,
                                       bool                       force_interleave = false);

template<typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t*                    processed_quantized_weight,
                        ComputeType*               scale_ptr,
                        WeightType const*          input_weight_ptr,
                        std::vector<size_t> const& shape,
                        QuantType                  quant_type,
                        bool                       force_interleave);

// This is exposed so that we can write tests that use the processed weights for CUTLASS but the unprocessed weight
// to implement a simple reference implementation.
template<typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t*                    processed_quantized_weight,
                        int8_t*                    unprocessed_quantized_weight,
                        ComputeType*               scale_ptr,
                        WeightType const*          input_weight_ptr,
                        std::vector<size_t> const& shape,
                        QuantType                  quant_type,
                        bool                       force_interleave);

}  // namespace lyradiff
