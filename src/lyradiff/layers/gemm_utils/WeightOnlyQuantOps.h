#include "src/lyrallms/utils/th_ops/th_utils.h"

namespace torch_ext {
using torch::Tensor;

std::vector<Tensor> symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, torch::ScalarType quant_type);
}  // namespace torch_ext