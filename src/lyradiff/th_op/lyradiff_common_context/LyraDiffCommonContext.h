#pragma once

#include "src/lyradiff/th_op/th_utils.h"
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/auth_utils.h"
#include "src/lyradiff/utils/cublasMMWrapper.h"
#include <cstdlib>
#include <cudnn.h>
#include <nvml.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace th = torch;
using std::vector;
using namespace lyradiff;
using namespace std;

namespace torch_ext {

class LyraDiffCommonContext: public th::jit::CustomClassHolder {
public:
    LyraDiffCommonContext();
    ~LyraDiffCommonContext();

    cudaStream_t stream_ = cudaStreamDefault;

    Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    cublasHandle_t                  cublas_handle_;
    cublasLtHandle_t                cublaslt_handle_;
    cudnnHandle_t                   cudnn_handle_;

    cublasAlgoMap*           cublas_algo_map_;
    std::mutex*              cublas_wrapper_mutex_;
    lyradiff::cublasMMWrapper* cublas_wrapper_;
};
}  // namespace torch_ext
