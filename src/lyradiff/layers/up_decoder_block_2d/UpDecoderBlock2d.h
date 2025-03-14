#pragma once

#include "UpDecoderBlock2dWeight.h"
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
class UpDecoderBlock2d: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        out_channels_;
    size_t        prev_output_channel_;
    size_t        norm_num_groups_;
    size_t        temb_channels_;
    cudnnHandle_t cudnn_handle_;

    bool add_upsample_;
    bool is_upcast_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width);

protected:
    Resnet2DBlock<T>* resnet2d_block;
    Resnet2DBlock<T>* resnet2d_block_pre;
    Conv2d<T>*        upsampler_conv;

public:
    T* hidden_state_buf_  = nullptr;
    T* hidden_state_buf2_ = nullptr;
    T* interpolate_buf_   = nullptr;

    UpDecoderBlock2d(const size_t     in_channels,
                     const size_t     out_channels,
                     const size_t     norm_num_groups,
                     const size_t     temb_channels,
                     cudnnHandle_t    cudnn_handle,
                     cudaStream_t     stream,
                     cublasMMWrapper* cublas_wrapper,
                     IAllocator*      allocator,
                     bool             is_free_buffer_after_forward,
                     bool             add_upsample,
                     bool             is_upcast = false);

    UpDecoderBlock2d(UpDecoderBlock2d<T> const& up_decoder_block2d);

    virtual ~UpDecoderBlock2d();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const UpDecoderBlock2dWeight<T>*   weights);
    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const UpDecoderBlock2dWeight<T>* weights);
};

}  // namespace lyradiff
