

#pragma once

#include "src/lyradiff/kernels/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {

std::vector<lyradiff::cutlass_extensions::CutlassGemmConfig> get_candidate_configs(int        sm,
                                                                                 bool const is_weight_only,
                                                                                 bool const simt_configs_only,
                                                                                 bool const int8_configs_only  = false,
                                                                                 int const  max_split_k        = 1,
                                                                                 bool const enable_hopper_gmma = false);

lyradiff::cutlass_extensions::CutlassGemmConfig estimate_best_config_from_occupancies(
    std::vector<lyradiff::cutlass_extensions::CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const&                                           occupancies,
    const int64_t                                                     m,
    const int64_t                                                     n,
    const int64_t                                                     k,
    const int64_t                                                     num_experts,
    int const                                                         split_k_limit,
    const size_t                                                      workspace_bytes,
    int const                                                         multi_processor_count,
    int const                                                         is_weight_only);

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
