#pragma once

#include "ControlNetConditioningEmbeddingWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class ControlNetConditioningEmbedding: public BaseLayer {
private:
    // params for 2 Identical ResNets
    std::vector<size_t> block_out_channels_ = {16, 32, 96, 256};
    size_t              conditioning_channels_;
    size_t              conditioning_embedding_channels_;

    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);
    void freeBuffer() override;

public:
    // resnet2d blocks
    std::vector<Conv2d<T>*> block_convs_ = std::vector<Conv2d<T>*>(6, nullptr);

    Conv2d<T>* input_conv_  = nullptr;
    Conv2d<T>* output_conv_ = nullptr;

    // buffers
    std::vector<T*> block_bufs_  = std::vector<T*>(6, nullptr);
    T*              conv_in_buf_ = nullptr;

    ControlNetConditioningEmbedding(const size_t     conditioning_channels,
                                    const size_t     conditioning_embedding_channels,
                                    cudnnHandle_t    cudnn_handle,
                                    cudaStream_t     stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator*      allocator,
                                    const bool       is_free_buffer_after_forward,
                                    const bool       sparse);

    ControlNetConditioningEmbedding(ControlNetConditioningEmbedding<T> const& other);

    virtual ~ControlNetConditioningEmbedding();

    virtual void forward(TensorMap*                                      output_tensors,
                         const TensorMap*                                input_tensors,
                         const ControlNetConditioningEmbeddingWeight<T>* weights);
};

}  // namespace lyradiff
