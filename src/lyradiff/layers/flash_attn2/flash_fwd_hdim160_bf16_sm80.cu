#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 160>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim160<cutlass::bfloat16_t>(params, stream);
}
