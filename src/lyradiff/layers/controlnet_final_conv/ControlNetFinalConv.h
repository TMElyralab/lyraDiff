#pragma once

#include "ControlNetFinalConvWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class ControlNetFinalConv: public BaseLayer {
private:
    // params for 2 Identical ResNets
    std::vector<size_t> block_out_channels_;
    // handler
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;

public:
    // resnet2d blocks
    std::vector<Conv2d<T>*> block_convs_;
    // Conv2d<T> *mid_block_conv_ = nullptr;

    ControlNetFinalConv(std::vector<size_t> block_out_channels,
                        cudnnHandle_t       cudnn_handle,
                        cudaStream_t        stream,
                        cublasMMWrapper*    cublas_wrapper,
                        IAllocator*         allocator,
                        const bool          is_free_buffer_after_forward,
                        const bool          sparse);

    ControlNetFinalConv(ControlNetFinalConv<T> const& other);

    virtual ~ControlNetFinalConv();

    virtual void forward(std::vector<Tensor>&                output_tensors,
                         const std::vector<Tensor>&          input_tensors,
                         const ControlNetFinalConvWeight<T>* weights,
                         const std::vector<float>&           controlnet_scales);
};

}  // namespace lyradiff
