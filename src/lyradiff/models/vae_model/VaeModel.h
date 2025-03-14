#pragma once

#include "VaeModelWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/vae_decoder/VaeDecoder.h"
#include "src/lyradiff/layers/vae_encoder/VaeEncoder.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class VaeModel: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t norm_num_groups_ = 32;
    size_t input_channels_  = 3;
    size_t output_channels_ = 3;
    size_t latent_channels_ = 4;

    size_t prev_batch  = 0;
    size_t prev_height = 0;
    size_t prev_width  = 0;

    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void decodeAllocateBuffer(size_t batch_size, size_t height, size_t width);
    void encodeAllocateBuffer(size_t batch_size, size_t height, size_t width);

public:
    T* post_quant_conv_buf_ = nullptr;
    T* encoder_res_buf_     = nullptr;

    Conv2d<T>*     post_quant_conv = nullptr;
    Conv2d<T>*     quant_conv      = nullptr;
    VaeDecoder<T>* vae_decoder     = nullptr;
    VaeEncoder<T>* vae_encoder     = nullptr;

    VaeModel(cudnnHandle_t    cudnn_handle,
             cudaStream_t     stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator*      allocator,
             const bool       is_free_buffer_after_forward,
             const bool       sparse,
             const bool       is_upcast = false);

    VaeModel(VaeModel<T> const& other);

    virtual ~VaeModel();

    virtual void
    decode(TensorMap* output_tensors, const TensorMap* input_tensors, const VaeModelWeight<T>* vae_weights);

    virtual void
    encode(TensorMap* output_tensors, const TensorMap* input_tensors, const VaeModelWeight<T>* vae_weights);
};
}  // namespace lyradiff
