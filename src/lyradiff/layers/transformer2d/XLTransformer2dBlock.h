#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlock.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerInt8Block.h"
#include "src/lyradiff/layers/conv2d/conv2d.h"
#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class XLTransformer2dBlock: public BaseLayer {
private:
    // block params
    size_t        in_channels_;
    size_t        head_num_;
    size_t        dim_per_head_;
    size_t        inner_dim_;
    size_t        norm_num_groups_;
    size_t        cross_attn_dim_;
    size_t        inner_trans_num_;
    cudnnHandle_t cudnn_handle_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t height, size_t width);

protected:
    std::vector<BasicTransformerBlock<T>*>     transblock_vec;
    std::vector<BasicTransformerInt8Block<T>*> transblock_int8_vec;
    // BasicTransformerBlock<T>* basic_transformer_block;

public:
    T*      norm_hidden_state_buf_ = nullptr;
    double* norm_cache_buf_        = nullptr;
    T*      linear_out_buf_        = nullptr;
    // T *linear_out_buf_1_ = nullptr;
    T* basic_transformer_res_buf_  = nullptr;
    T* basic_transformer_res_buf_1 = nullptr;
    // T *linear_out_buf_2_ = nullptr;

    XLTransformer2dBlock(size_t           in_channels,
                         size_t           head_num,
                         size_t           dim_per_head,
                         size_t           cross_attn_dim,
                         size_t           norm_num_groups,
                         size_t           inner_trans_num,
                         cudnnHandle_t    cudnn_handle,
                         cudaStream_t     stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator*      allocator,
                         bool             is_free_buffer_after_forward,
                         LyraQuantType    quant_level = LyraQuantType::NONE);

    XLTransformer2dBlock(XLTransformer2dBlock<T> const& xltransformer2dBlock);

    virtual ~XLTransformer2dBlock();

    virtual void forward(std::vector<lyradiff::Tensor>*         output_tensors,
                         const std::vector<lyradiff::Tensor>*   input_tensors,
                         const XLTransformer2dBlockWeight<T>* weights);
    virtual void
    forward(TensorMap* output_tensors, const TensorMap* input_tensors, const XLTransformer2dBlockWeight<T>* weights);
};

}  // namespace lyradiff
