#pragma once

#include "3rdparty/INIReader.h"
#include "cuda_fp8_utils.h"
#include "lyraException.h"
#include "src/lyradiff/common.h"
#include "src/lyradiff/utils/logger.h"
#include <assert.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef SPARSITY_ENABLED
#include <cusparseLt.h>
#endif

namespace lyradiff {

#define MAX_CONFIG_NUM 20
#define COL32_ 32
// workspace for cublas gemm : 64MB
#define CUBLAS_WORKSPACE_SIZE 67108864

/* **************************** type definition ***************************** */

enum CublasDataType {
    FLOAT_DATATYPE    = 0,
    HALF_DATATYPE     = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE     = 3,
    FP8_DATATYPE      = 4
};

enum FtCudaDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8  = 4
};

enum LyraQuantType {
    NONE                     = 0,
    FP8_W8A8                 = 1,
    FP8_W8A8_FULL            = 2,
    FP8_WEIGHT_ONLY          = 3,
    INT8_SMOOTH_QUANT_LEVEL1 = 4,
    INT8_SMOOTH_QUANT_LEVEL2 = 5,
    INT8_SMOOTH_QUANT_LEVEL3 = 6,
    INT4_W4A4                = 7,
    INT4_W4A4_FULL           = 8,
};

/* **************************** debug tools ********************************* */

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline void syncAndCheck(const char* const file, int const line)
{
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("lyradiff_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[lyradiff][ERROR] CUDA runtime error: ")
                                         + (_cudaGetErrorEnum(result)) + " " + file + ":" + std::to_string(line)
                                         + " \n");
            }
            FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[lyradiff][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

#define checkCUDNN(expression)                                                                                         \
    {                                                                                                                  \
        cudnnStatus_t status = (expression);                                                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                          \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(status)   \
                      << std::endl;                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    }

template<typename T>
void print_to_file(const T*           result,
                   const int          size,
                   const char*        file,
                   cudaStream_t       stream    = 0,
                   std::ios::openmode open_mode = std::ios::out);

template<typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream, std::string name = "");

template<typename T>
void print_to_screen(const T* result, const int size);

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void check_max_val(const T* result, const int size);

template<typename T>
void check_abs_mean_val(const T* result, const int size);

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw LyraException(file, line, fmtstr("[lyradiff][ERROR] Assertion failed: %s", info.c_str()));
}

#if defined(_WIN32)
#define LYRA_LIKELY(x) (__assume((x) == 1), (x))
#else
#define LYRA_LIKELY(x) __builtin_expect((x), 1)
#endif

#define LYRA_CHECK(val)                                                                                                \
    do {                                                                                                               \
        LYRA_LIKELY(static_cast<bool>(val)) ? ((void)0) : lyradiff::throwRuntimeError(__FILE__, __LINE__, #val);         \
    } while (0)

#define LYRA_CHECK_WITH_INFO(val, info)                                                                                \
    do {                                                                                                               \
        LYRA_LIKELY(static_cast<bool>(val)) ? ((void)0) : lyradiff::throwRuntimeError(__FILE__, __LINE__, info);         \
    } while (0)

#define LYRA_THROW(...)                                                                                                \
    do {                                                                                                               \
        throw NEW_LYRA_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define LYRA_WRAP(ex)                                                                                                  \
    NEW_LYRA_EXCEPTION("%s: %s", lyradiff::LyraException::demangle(typeid(ex).name()).c_str(), ex.what())

#ifdef SPARSITY_ENABLED
#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            throw std::runtime_error(std::string("[lyradiff][ERROR] CUSPARSE API failed at line ")                       \
                                     + std::to_string(__LINE__) + " in file " + __FILE__ + ": "                        \
                                     + cusparseGetErrorString(status) + " " + std::to_string(status));                 \
        }                                                                                                              \
    }
#endif

/*************Time Handling**************/
class CudaTimer {
private:
    cudaEvent_t  event_start_;
    cudaEvent_t  event_stop_;
    cudaStream_t stream_;

public:
    explicit CudaTimer(cudaStream_t stream = 0)
    {
        stream_ = stream;
    }
    void start()
    {
        check_cuda_error(cudaEventCreate(&event_start_));
        check_cuda_error(cudaEventCreate(&event_stop_));
        check_cuda_error(cudaEventRecord(event_start_, stream_));
    }
    float stop()
    {
        float time;
        check_cuda_error(cudaEventRecord(event_stop_, stream_));
        check_cuda_error(cudaEventSynchronize(event_stop_));
        check_cuda_error(cudaEventElapsedTime(&time, event_start_, event_stop_));
        check_cuda_error(cudaEventDestroy(event_start_));
        check_cuda_error(cudaEventDestroy(event_stop_));
        return time;
    }
    ~CudaTimer() {}
};

static double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline void print_mem_usage(std::string time = "after allocation")
{
    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getMaxSharedMemoryPerBlock()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int max_shared_memory_size = 0;
    check_cuda_error(cudaDeviceGetAttribute(&max_shared_memory_size, cudaDevAttrMaxSharedMemoryPerBlock, device));
    return max_shared_memory_size;
}

inline std::string getDeviceName()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    return std::string(props.name);
}

inline int div_up(int a, int n)
{
    return (a + n - 1) / n;
}

cudaError_t getSetDevice(int i_device, int* o_device = NULL);

inline int getDevice()
{
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount()
{
    int count = 0;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
}

template<typename T>
CublasDataType getCublasDataType()
{
    if (std::is_same<T, half>::value) {
        return HALF_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return BFLOAT16_DATATYPE;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return FLOAT_DATATYPE;
    }
    else {
        LYRA_CHECK(false);
        return FLOAT_DATATYPE;
    }
}

template<typename T>
cudaDataType_t getCudaDataType()
{
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    }
    else {
        LYRA_CHECK(false);
        return CUDA_R_32F;
    }
}

template<CublasDataType T>
struct getTypeFromCudaDataType {
    using Type = float;
};

template<>
struct getTypeFromCudaDataType<HALF_DATATYPE> {
    using Type = half;
};

#ifdef ENABLE_BF16
template<>
struct getTypeFromCudaDataType<BFLOAT16_DATATYPE> {
    using Type = __nv_bfloat16;
};
#endif

FtCudaDataType getModelFileType(std::string ini_file, std::string section_name);

// clang-format off
template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif

// inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
// inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
// clang-format on

template<typename T1, typename T2>
void compareTwoTensor(
    const T1* pred, const T2* ref, const int size, const int print_size = 0, const std::string filename = "")
{
    T1* h_pred = new T1[size];
    T2* h_ref  = new T2[size];
    check_cuda_error(cudaMemcpy(h_pred, pred, size * sizeof(T1), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_ref, ref, size * sizeof(T2), cudaMemcpyDeviceToHost));

    FILE* fd = nullptr;
    if (filename != "") {
        fd = fopen(filename.c_str(), "w");
        fprintf(fd, "| %10s | %10s | %10s | %10s | \n", "pred", "ref", "abs_diff", "rel_diff(%)");
    }

    if (print_size > 0) {
        FT_LOG_INFO("  id |   pred  |   ref   |abs diff | rel diff (%) |");
    }
    float mean_abs_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    int   count         = 0;
    for (int i = 0; i < size; i++) {
        if (i < print_size) {
            FT_LOG_INFO("%4d | % 6.4f | % 6.4f | % 6.4f | % 7.4f |",
                        i,
                        (float)h_pred[i],
                        (float)h_ref[i],
                        abs((float)h_pred[i] - (float)h_ref[i]),
                        abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
        if ((float)h_pred[i] == 0) {
            continue;
        }
        count += 1;
        mean_abs_diff += abs((float)h_pred[i] - (float)h_ref[i]);
        mean_rel_diff += abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f;

        if (fd != nullptr) {
            fprintf(fd,
                    "| %10.5f | %10.5f | %10.5f | %11.5f |\n",
                    (float)h_pred[i],
                    (float)h_ref[i],
                    abs((float)h_pred[i] - (float)h_ref[i]),
                    abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
    }
    mean_abs_diff = mean_abs_diff / (float)count;
    mean_rel_diff = mean_rel_diff / (float)count;
    FT_LOG_INFO("mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);

    if (fd != nullptr) {
        fprintf(fd, "mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);
        fclose(fd);
    }
    delete[] h_pred;
    delete[] h_ref;
}

/* ************************** end of common utils ************************** */
}  // namespace lyradiff
