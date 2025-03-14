#include "LyraDiffCommonContext.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace th = torch;
namespace torch_ext {

LyraDiffCommonContext::LyraDiffCommonContext()
{
    at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    stream_                            = defaultStream.stream();
    allocator_                         = new Allocator<AllocatorType::CUDA>(getDevice());

    cublas_algo_map_ = new cublasAlgoMap(GEMM_CONFIG);

    cudnnCreate(&cudnn_handle_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);

    cublas_wrapper_mutex_ = new std::mutex();

    cublas_wrapper_ = new lyradiff::cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
}

LyraDiffCommonContext::~LyraDiffCommonContext()
{
    allocator_->freeAllNameBuf();
    cudaDeviceSynchronize();

    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
    delete cublas_wrapper_;
    delete allocator_;

    cublasLtDestroy(cublaslt_handle_);
    cublasDestroy(cublas_handle_);
    cudnnDestroy(cudnn_handle_);
}

}  // namespace torch_ext
