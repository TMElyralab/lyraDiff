#pragma once

#include "VaeDecoderWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/unet_mid_block_2d/UNetMidBlock2D.h"
#include "src/lyradiff/layers/up_decoder_block_2d/UpDecoderBlock2d.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class VaeDecoder: public BaseLayer {
private:
    // block params
    std::vector<size_t> block_out_channels_ = {512, 512, 256, 128};
    size_t              temb_channels_      = 0;
    size_t              in_channels_;
    size_t              out_channels_;
    size_t              norm_num_groups_;
    cudnnHandle_t       cudnn_handle_;

    size_t prev_batch  = 0;
    size_t prev_height = 0;
    size_t prev_width  = 0;

    bool is_upcast_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);

protected:
    Conv2d<T>* conv_in  = nullptr;
    Conv2d<T>* conv_out = nullptr;

    UNetMidBlock2D<T>*   mid_block;
    UpDecoderBlock2d<T>* up_decoder_block_0;
    UpDecoderBlock2d<T>* up_decoder_block_1;
    UpDecoderBlock2d<T>* up_decoder_block_2;
    UpDecoderBlock2d<T>* up_decoder_block_3;

    UpDecoderBlock2d<float>* upcast_up_decoder_block_2;
    UpDecoderBlock2d<float>* upcast_up_decoder_block_3;

public:
    T* mid_block_buf_      = nullptr;
    T* up_block_share_buf_ = nullptr;
    T* gnorm_buf_          = nullptr;

    float* upcast_up_block_share_buf_ = nullptr;
    float* upcast_gnorm_buf_          = nullptr;
    double* gnorm_cache_               = nullptr;

    T* conv_buf_ = nullptr;

    VaeDecoder(const size_t     in_channels,
               const size_t     out_channels,
               const size_t     norm_num_groups,
               cudnnHandle_t    cudnn_handle,
               cudaStream_t     stream,
               cublasMMWrapper* cublas_wrapper,
               IAllocator*      allocator,
               bool             is_free_buffer_after_forward,
               const bool       is_upcast = false);

    VaeDecoder(VaeDecoder<T> const& other);

    virtual ~VaeDecoder();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const VaeDecoderWeight<T>*         weights);
    virtual void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const VaeDecoderWeight<T>* weights);
};

}  // namespace lyradiff
