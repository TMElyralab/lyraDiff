#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlockWeight.h"
#include "src/lyradiff/layers/time_proj/TimeProjection.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlock.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class TextTimeEmbeddingBlock: public BaseLayer {
private:
    // params
    size_t time_proj_out_dim_;
    size_t augemb_time_proj_out_dim_;
    size_t text_emb_dim_;
    size_t temb_channels_;
    size_t augemb_temb_input_dim_;

    // buffer
    T* temb_buffer_               = nullptr;
    T* augemb_temb_buffer_        = nullptr;
    T* output_buffer_             = nullptr;
    T* augemb_cat_tproj_text_buf_ = nullptr;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t outputBytes);

public:
    TimeProjection<T>*         augemb_time_proj = nullptr;
    TimeProjection<T>*         time_proj        = nullptr;
    TimestepEmbeddingBlock<T>* time_emb         = nullptr;
    TimestepEmbeddingBlock<T>* augemb_temb      = nullptr;

    TextTimeEmbeddingBlock(const size_t     time_proj_out_dim,
                           const size_t     augemb_time_proj_out_dim,
                           const size_t     text_emb_dim,
                           const size_t     temb_channels,
                           cudaStream_t     stream,
                           cublasMMWrapper* cublas_wrapper,
                           IAllocator*      allocator,
                           const bool       is_free_buffer_after_forward,
                           const bool       sparse);

    TextTimeEmbeddingBlock(TextTimeEmbeddingBlock<T> const& text_temb_block);

    virtual ~TextTimeEmbeddingBlock();

    virtual void forward(TensorMap*                             output_tensors,
                         const TensorMap*                       input_tensors,
                         const float                            timestep,
                         const TextTimeEmbeddingBlockWeight<T>* text_timeembedding_block_weight);
};

}  // namespace lyradiff
