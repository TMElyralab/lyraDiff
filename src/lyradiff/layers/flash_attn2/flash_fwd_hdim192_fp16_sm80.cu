#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream)
{
    run_mha_fwd_hdim192<cutlass::half_t>(params, stream);
}
