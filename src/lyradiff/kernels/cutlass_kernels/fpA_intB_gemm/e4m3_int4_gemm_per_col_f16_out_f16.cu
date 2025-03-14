

#include "src/lyradiff/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace lyradiff
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_FP8
template class CutlassFpAIntBGemmRunner<__nv_fp8_e4m3,       
    cutlass::int4b_t,                                        
    cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, half, 
    half,                                                    
    half                                                     
    >;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace lyradiff
