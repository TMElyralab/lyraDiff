

#include "src/lyradiff/kernels/cutlass_kernels/gemm/gemm_template.h"

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {

template class CutlassGemmRunner<float>;  // for compilation only

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
