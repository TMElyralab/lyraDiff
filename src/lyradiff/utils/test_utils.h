#pragma once

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace lyradiff {

namespace PerfTime {
// 耗时统计
enum class PerfType {
    PERF_E2E = 0,  // 端到端类型，支持多线程
    PERF_ACC       // 累加类型，支持多线程
};

void StartHit(PerfType type, const char* mark);                // 开始打点
void EndHit(PerfType type, const char* mark);                  // 结束打点
void OutputReport(const char* file, bool isOutputCmd = true);  // 输出报告

}  // namespace PerfTime

#define TIMEIT(print, n, stream, fn, ...)                                                                              \
    ({                                                                                                                 \
        cudaEvent_t _macro_event_start, _macro_event_stop;                                                             \
        cudaEventCreate(&_macro_event_start);                                                                          \
        cudaEventCreate(&_macro_event_stop);                                                                           \
        cudaEventRecord(_macro_event_start, stream);                                                                   \
        for (int i = 0; i < n; i++) {                                                                                  \
            fn(__VA_ARGS__);                                                                                           \
        }                                                                                                              \
        cudaEventRecord(_macro_event_stop, stream);                                                                    \
        cudaStreamSynchronize(stream);                                                                                 \
        float ms = 0.0f;                                                                                               \
        cudaEventElapsedTime(&ms, _macro_event_start, _macro_event_stop);                                              \
        ms /= n;                                                                                                       \
        if (print)                                                                                                     \
            printf("[TIMEIT] " #fn ": %.2fµs\n", ms * 1000);                                                           \
        ms;                                                                                                            \
    })

template<typename T>
struct rel_abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs == 0 ? 0 : static_cast<T>(fabs(lhs - rhs) / fabs(lhs));
    }
};

template<typename T>
struct abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return static_cast<T>(fabs(lhs - rhs));
    }
};

}  // namespace lyradiff
