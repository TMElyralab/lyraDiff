#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlock.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerInt8Block.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/transformer2d/Transformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class Transformer2dBlock: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        head_num_;
    size_t        dim_per_head_;
    size_t        inner_dim_;
    size_t        norm_num_groups_;
    size_t        cross_attn_dim_;
    cudnnHandle_t cudnn_handle_;
    bool          use_linear_projection_ = true;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);

protected:
    BasicTransformerBlock<T>*     basic_transformer_block;
    BasicTransformerInt8Block<T>* basic_transformer_int8_block;
    Conv2d<T>*                    conv1;
    Conv2d<T>*                    conv2;

public:
    T*      norm_hidden_state_buf_     = nullptr;
    double* norm_cache_buf_            = nullptr;
    T*      conv_out_buf_              = nullptr;
    T*      basic_transformer_res_buf_ = nullptr;

    Transformer2dBlock(size_t           in_channels,
                       size_t           head_num,
                       size_t           dim_per_head,
                       size_t           cross_attn_dim,
                       size_t           norm_num_groups,
                       cudnnHandle_t    cudnn_handle,
                       cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       LyraQuantType    quant_level = LyraQuantType::NONE);

    Transformer2dBlock(Transformer2dBlock<T> const& transformer2dBlock);

    virtual ~Transformer2dBlock();

    virtual void forward(std::vector<lyradiff::Tensor>*       output_tensors,
                         const std::vector<lyradiff::Tensor>* input_tensors,
                         const Transformer2dBlockWeight<T>* weights);
    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const Transformer2dBlockWeight<T>* weights);
};

}  // namespace lyradiff
