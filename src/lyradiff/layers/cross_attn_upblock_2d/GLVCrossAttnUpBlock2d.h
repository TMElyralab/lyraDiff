#pragma once

#include "GLVCrossAttnUpBlock2dWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/resnet2d/Resnet2DBlock.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlock.h"
#include "src/lyradiff/layers/zero_cross_attn/ZeroCrossAttn.h"
#include "src/lyradiff/layers/zero_sft/ZeroSFT.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class GLVCrossAttnUpBlock2d: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        out_channels_;
    size_t        prev_output_channel_;
    size_t        temb_channels_;
    size_t        head_num_;
    size_t        norm_num_groups_;
    size_t        cross_attn_dim_;
    size_t        inner_trans_num_;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t  stream_assistant_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width);

protected:
    XLTransformer2dBlock<T>* xltransformer2d_block;
    Resnet2DBlock<T>*        resnet2d_block1;
    Resnet2DBlock<T>*        resnet2d_block2;
    Resnet2DBlock<T>*        resnet2d_block3;
    Conv2d<T>*               upsampler_conv;

    ZeroSFT<T>* project_module1 = nullptr;
    ZeroSFT<T>* project_module2 = nullptr;
    ZeroSFT<T>* project_module3 = nullptr;

    ZeroCrossAttn<T>* cross_project_module = nullptr;

public:
    T* hidden_state_buf1_ = nullptr;
    T* hidden_state_buf2_ = nullptr;
    T* cat_buf1_          = nullptr;
    T* cat_buf2_          = nullptr;
    T* cat_buf3_          = nullptr;
    T* interpolate_buf_   = nullptr;

    GLVCrossAttnUpBlock2d(const size_t     in_channels,
                          const size_t     out_channels,
                          const size_t     prev_output_channel,
                          const size_t     temb_channels,
                          const size_t     head_num,
                          const size_t     cross_attn_dim,
                          const size_t     norm_num_groups,
                          const size_t     inner_trans_num,
                          cudnnHandle_t    cudnn_handle,
                          cudaStream_t     stream,
                          cudaStream_t     stream_assistant,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward,
                          LyraQuantType    quant_level = LyraQuantType::NONE);

    GLVCrossAttnUpBlock2d(GLVCrossAttnUpBlock2d<T> const& cross_attn_up_block2d);

    virtual ~GLVCrossAttnUpBlock2d();

    virtual void forward(std::vector<lyradiff::Tensor>*          output_tensors,
                         const std::vector<lyradiff::Tensor>*    input_tensors,
                         const GLVCrossAttnUpBlock2dWeight<T>* weights,
                         const float                           control_scale = 0.0);
    virtual void forward(TensorMap*                            output_tensors,
                         const TensorMap*                      input_tensors,
                         const GLVCrossAttnUpBlock2dWeight<T>* weights,
                         const float                           control_scale = 0.0);
};

}  // namespace lyradiff
