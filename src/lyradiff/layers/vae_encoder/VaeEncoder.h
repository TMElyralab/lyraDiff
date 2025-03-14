#pragma once

#include "VaeEncoderWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/down_encoder_block_2d/DownEncoderBlock2D.h"
#include "src/lyradiff/layers/unet_mid_block_2d/UNetMidBlock2D.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class VaeEncoder: public BaseLayer {
private:
    // block params
    std::vector<size_t> block_out_channels_ = {128, 128, 256, 512};
    size_t              temb_channels_      = 0;
    size_t              in_channels_;
    size_t              out_channels_;
    size_t              norm_num_groups_;
    cudnnHandle_t       cudnn_handle_;

    size_t prev_batch  = 0;
    size_t prev_height = 0;
    size_t prev_width  = 0;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);

protected:
    Conv2d<T>* conv_in  = nullptr;
    Conv2d<T>* conv_out = nullptr;

    UNetMidBlock2D<T>*     mid_block;
    DownEncoderBlock2D<T>* down_encoder_block_0;
    DownEncoderBlock2D<T>* down_encoder_block_1;
    DownEncoderBlock2D<T>* down_encoder_block_2;
    DownEncoderBlock2D<T>* down_encoder_block_3;

public:
    T*      mid_block_buf_        = nullptr;
    T*      down_block_share_buf_ = nullptr;
    T*      gnorm_buf_            = nullptr;
    double* gnorm_cache_          = nullptr;

    VaeEncoder(const size_t     in_channels,
               const size_t     out_channels,
               const size_t     norm_num_groups,
               cudnnHandle_t    cudnn_handle,
               cudaStream_t     stream,
               cublasMMWrapper* cublas_wrapper,
               IAllocator*      allocator,
               bool             is_free_buffer_after_forward);

    VaeEncoder(VaeEncoder<T> const& other);

    virtual ~VaeEncoder();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const VaeEncoderWeight<T>*         weights);
    virtual void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const VaeEncoderWeight<T>* weights);
};

}  // namespace lyradiff
