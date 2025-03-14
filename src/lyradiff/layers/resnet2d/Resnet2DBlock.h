#pragma once

#include "Resnet2DBlockWeight.h"
#include "math.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/kernels/resnet2d/resnet2d_kernels.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/cublasMMWrapper.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/string_utils.h"
#include <cuda_runtime.h>
#include <numeric>

namespace lyradiff {

template<typename T>
class Resnet2DBlock: public BaseLayer {
private:
    // params
    size_t in_channels_;
    size_t out_channels_;
    size_t ngroups_in_;
    size_t ngroups_out_;
    bool   use_swish_;
    bool   conv_shortcut_;
    bool   has_temb_;
    size_t time_emb_in_dim_;
    // weights

    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    // inner buffers
    double* gnorm1_caches_ = nullptr;
    double* gnorm2_caches_ = nullptr;

    T* temb_buf_1 = nullptr;
    T* temb_buf_2 = nullptr;

    T* inner_buf_1    = nullptr;
    T* inner_conv_buf = nullptr;

    Conv2d<T>* input_conv_    = nullptr;
    Conv2d<T>* second_conv_   = nullptr;
    Conv2d<T>* shortcut_conv_ = nullptr;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(const size_t batch_size, const size_t input_bytes, const size_t output_bytes);

public:
    Resnet2DBlock(const size_t     in_channels,
                  const size_t     out_channels,
                  const size_t     ngroups_in,
                  const size_t     ngroups_out,
                  const bool       use_swish,
                  cudnnHandle_t    cudnn_handle,
                  cudaStream_t     stream_main,
                  cudaStream_t     stream_assistant,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  bool             is_free_buffer_after_forward,
                  bool             has_temb = true);

    Resnet2DBlock(const size_t     in_channels,
                  const size_t     out_channels,
                  const size_t     ngroups_in,
                  const size_t     ngroups_out,
                  const bool       use_swish,
                  const size_t     time_emb_in_dim,
                  cudnnHandle_t    cudnn_handle,
                  cudaStream_t     stream_main,
                  cudaStream_t     stream_assistant,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  bool             is_free_buffer_after_forward,
                  bool             has_temb = true);

    Resnet2DBlock(Resnet2DBlock<T> const& resnet2DBlock);

    virtual ~Resnet2DBlock();
    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const Resnet2DBlockWeight<T>*      weights);
    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const Resnet2DBlockWeight<T>* weights);
};

}  // namespace lyradiff