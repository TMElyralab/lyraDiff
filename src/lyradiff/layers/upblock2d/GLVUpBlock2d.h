#pragma once

#include "GLVUpBlock2dWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/zero_sft/ZeroSFT.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class GLVUpBlock2d: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        out_channels_;
    size_t        prev_output_channel_;
    size_t        norm_num_groups_;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size,
                        size_t height,
                        size_t width,
                        size_t hidden_channel,
                        size_t target_height,
                        size_t target_width);

protected:
    Resnet2DBlock<T>* resnet2d_block0;
    Resnet2DBlock<T>* resnet2d_block;

    ZeroSFT<T>* project_module1 = nullptr;
    ZeroSFT<T>* project_module2 = nullptr;
    // ZeroSFT<T>* project_module3 = nullptr;

public:
    T* cat_buf_   = nullptr;
    T* cat_buf_0_ = nullptr;

    GLVUpBlock2d(const size_t     in_channels,
                 const size_t     out_channels,
                 const size_t     prev_output_channel,
                 const size_t     norm_num_groups,
                 cudnnHandle_t    cudnn_handle,
                 cudaStream_t     stream,
                 cudaStream_t     stream_assistant,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward);

    GLVUpBlock2d(GLVUpBlock2d<T> const& up_block2d);

    virtual ~GLVUpBlock2d();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const GLVUpBlock2dWeight<T>*       weights,
                         const float                        control_scale = 0.0);

    virtual void forward(TensorMap*                   output_tensors,
                         const TensorMap*             input_tensors,
                         const GLVUpBlock2dWeight<T>* weights,
                         const float                  control_scale = 0.0);
};

}  // namespace lyradiff
