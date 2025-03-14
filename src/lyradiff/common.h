#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace lyradiff {

// 数据操作类型的枚举类，暂定 FP32 和 FP16==HALF
enum class OperationType {
    FP32,
    HALF
};

// 模板化操作类型的 Traits 类，
// Traits 用于封装 CUDA 和 CUBLAS 常用参数
template<OperationType OpType>
class Traits;

// FP32 Traits 具象化
template<>
class Traits<OperationType::FP32> {
public:
    typedef float DataType;
    // cuBLAS parameters
    static cudaDataType_t const computeType = CUDA_R_32F;
    static cudaDataType_t const AType       = CUDA_R_32F;
    static cudaDataType_t const BType       = CUDA_R_32F;
    static cudaDataType_t const CType       = CUDA_R_32F;

    // 在 cuBLAS 库中，`cublasGemmEx()` 函数的 `algo` 参数用于指定矩阵乘法的算法。

    // `algo` 参数的取值是一个枚举类型，包括以下几种：

    // - `CUBLAS_GEMM_DFALT`：默认算法，由 cuBLAS 库根据硬件和输入矩阵大小自动选择最优算法。
    // - `CUBLAS_GEMM_ALGO0`：使用 SM 版本 1.x 的算法。
    // - `CUBLAS_GEMM_ALGO1`：使用 SM 版本 1.x 的算法，对于大矩阵有更好的性能。
    // - `CUBLAS_GEMM_ALGO2`：使用 SM 版本 2.x 的算法。
    // - `CUBLAS_GEMM_ALGO3`：使用 SM 版本 2.x 的算法，对于大矩阵有更好的性能。
    // - `CUBLAS_GEMM_ALGO4`：使用 SM 版本 3.x 的算法。
    // - `CUBLAS_GEMM_ALGO5`：使用 SM 版本 3.x 的算法，对于大矩阵有更好的性能。
    // - `CUBLAS_GEMM_ALGO6`：使用 SM 版本 6.x 的算法。
    // - `CUBLAS_GEMM_ALGO7`：使用 SM 版本 6.x 的算法，对于大矩阵有更好的性能。

    // 不同的算法对于不同的硬件和矩阵大小有不同的性能表现。一般来说，cuBLAS 库的默认算法已经足够优秀，
    // 因此在使用 `cublasGemmEx()` 函数时，可以直接使用默认算法，即 `CUBLAS_GEMM_DFALT`。
    // 如果需要进一步优化性能，可以尝试使用其他算法，并根据实际情况进行选择。

    static const int algo = -1;
};

// FP16 Traits 具象化
template<>
class Traits<OperationType::HALF> {
public:
    typedef __half DataType;
    // cuBLAS parameters
    static cudaDataType_t const computeType = CUDA_R_16F;
    static cudaDataType_t const AType       = CUDA_R_16F;
    static cudaDataType_t const BType       = CUDA_R_16F;
    static cudaDataType_t const CType       = CUDA_R_16F;
    static const int            algo        = 99;
};

// 激活枚举
enum ActType {
    Relu,
    Sigmoid,
    SoftPlus,
    No
};

// 激活函数
template<ActType act, typename T>
__inline__ __device__ T act_fun(T val)
{
    if (act == ActType::Relu)
        return (val <= (T)0.0f) ? (T)0.0f : val;
    else if (act == ActType::SoftPlus)
        return __logf(__expf((float)val) + 1.0f);
    else if (act == ActType::Sigmoid)
        return 1.0f / (1.0f + __expf(-1.0f * (float)val));
    else
        return val;
}

// half4 自定义类型
// typedef union half4 {
//     float2 x;
//     half2  h[2];
// } half4;

// 宏函数和常用的 cuda 检验辅助函数
/**/
#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[BT][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
        throw std::runtime_error(std::string("[lyradiff][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

#define CUTLASS_CHECK(status)                                                                                          \
    {                                                                                                                  \
        ::cutlass::Status error = status;                                                                              \
        if (error != ::cutlass::Status::kSuccess) {                                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

}  // namespace lyradiff