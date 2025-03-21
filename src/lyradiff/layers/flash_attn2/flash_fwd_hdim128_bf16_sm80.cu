#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim128<cutlass::bfloat16_t>(params, stream);
}