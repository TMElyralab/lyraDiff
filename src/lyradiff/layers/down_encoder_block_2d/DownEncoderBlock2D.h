#pragma once

#include "DownEncoderBlock2DWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class DownEncoderBlock2D: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        out_channels_;
    size_t        prev_output_channel_;
    size_t        norm_num_groups_;
    size_t        temb_channels_;
    cudnnHandle_t cudnn_handle_;

    bool add_downsample_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width);

protected:
    Resnet2DBlock<T>* resnet2d_block_1;
    Resnet2DBlock<T>* resnet2d_block_2;
    Conv2d<T>*        downsampler_conv;

public:
    T* hidden_state_buf_  = nullptr;
    T* hidden_state_buf2_ = nullptr;
    T* pad_buf_ = nullptr;

    DownEncoderBlock2D(const size_t     in_channels,
                       const size_t     out_channels,
                       const size_t     norm_num_groups,
                       const size_t     temb_channels,
                       cudnnHandle_t    cudnn_handle,
                       cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       bool             add_downsample);

    DownEncoderBlock2D(DownEncoderBlock2D<T> const& other);

    virtual ~DownEncoderBlock2D();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const DownEncoderBlock2DWeight<T>*       weights);
    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const DownEncoderBlock2DWeight<T>* weights);
};

}  // namespace lyradiff
