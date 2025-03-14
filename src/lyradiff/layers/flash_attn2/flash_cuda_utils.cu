#include "flash_cuda_utils.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace at {
namespace cuda {
static std::once_flag                               g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<cudaDeviceProp>                  g_device_props;

static int GetCurrentDeviceId()
{
    int device_id;
    C10_CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

static int GetCudaDeviceCount()
{
    int count;
    C10_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

cudaDeviceProp* getCurrentDeviceProperties(int id)
{
    std::call_once(g_device_props_size_init_flag, [&] {
        int gpu_num = 0;
        gpu_num     = GetCudaDeviceCount();
        g_device_props_init_flags.resize(gpu_num);
        g_device_props.resize(gpu_num);
        for (int i = 0; i < gpu_num; ++i) {
            g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
        }
    });

    if (id == -1) {
        id = GetCurrentDeviceId();
    }

    std::call_once(*(g_device_props_init_flags[id]),
                   [&] { C10_CUDA_CHECK(cudaGetDeviceProperties(&g_device_props[id], id)); });

    return &g_device_props[id];
}
}  // namespace cuda
}  // namespace at