
#pragma once

#include <cuda_runtime_api.h>

#include "cutlass/device_kernel.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

namespace cutlass_extensions {

template<typename GemmKernel, bool enable_cutlass_3x = false>
inline int compute_occupancy_for_kernel()
{

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size > (48 << 10)) {
        cudaFuncAttributes attr;
        int                device             = 0;
        int                max_smem_per_block = 0;
        check_cuda_error(cudaGetDevice(&device));
        check_cuda_error(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if constexpr (enable_cutlass_3x) {
            check_cuda_error(cudaFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>));
        }
        else {
            check_cuda_error(cudaFuncGetAttributes(&attr, cutlass::Kernel<GemmKernel>));
        }
        if (smem_size + attr.sharedSizeBytes >= static_cast<size_t>(max_smem_per_block)) {
            // This should mean that
            // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
            // wouldn't work. In that case, we return an occupancy of 0. This will cause the heuristic to ignore this
            // configuration.
            return 0;
        }
    }

    int max_active_blocks = -1;
    if constexpr (enable_cutlass_3x) {
        check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            cutlass::device_kernel<GemmKernel>,
            128 * (GemmKernel::NumLoadWarpGroups + GemmKernel::NumMmaWarpGroups),
            smem_size));
    }
    else {
        check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));
    }

    return max_active_blocks;
}

}  // namespace cutlass_extensions
}  // namespace lyradiff
