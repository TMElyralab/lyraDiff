#pragma once

#include "ZeroSFTWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class ZeroSFT: public BaseLayer {
private:
    size_t cond_channels_;
    size_t project_channels_;
    size_t concat_channels_;
    size_t nhidden          = 128;
    size_t norm_num_groups_ = 32;

    cudnnHandle_t cudnn_handle_;

    bool is_mid_block_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);

public:
    Conv2d<T>* mlp_conv  = nullptr;
    Conv2d<T>* zero_add  = nullptr;
    Conv2d<T>* zero_mul  = nullptr;
    Conv2d<T>* zero_conv = nullptr;

    T* hidden_state_raw_buf_ = nullptr;
    T* hidden_state_buf1_    = nullptr;
    T* hidden_state_buf2_    = nullptr;
    T* actv_buf_             = nullptr;
    T* gamma_buf_            = nullptr;
    T* beta_buf_             = nullptr;
    T* gnorm_res_buf_        = nullptr;

    double* norm_cache_buf_ = nullptr;

    ZeroSFT(size_t           project_channels,
            size_t           cond_channels,
            size_t           concat_channels,
            bool             is_mid_block,
            cudnnHandle_t    cudnn_handle,
            cudaStream_t     stream,
            cublasMMWrapper* cublas_wrapper,
            IAllocator*      allocator,
            const bool       is_free_buffer_after_forward,
            const bool       sparse);

    ZeroSFT(ZeroSFT<T> const& other);

    virtual ~ZeroSFT();

    virtual void forward(TensorMap*              output_tensors,
                         TensorMap*              input_tensors,
                         const ZeroSFTWeight<T>* weights,
                         float                   control_scale = 0);
};

}  // namespace lyradiff
