#include "src/lyradiff/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {
#ifdef ENABLE_BF16
template class CutlassFpAIntBGemmRunner<__nv_bfloat16,
                                        cutlass::uint4b_t,
                                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;
#endif
}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
