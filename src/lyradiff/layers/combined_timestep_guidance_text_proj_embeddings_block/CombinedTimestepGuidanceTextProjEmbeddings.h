#pragma once

#include "CombinedTimestepGuidanceTextProjEmbeddingsWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/time_proj/TimeProjection.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlock.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class CombinedTimestepGuidanceTextProjEmbeddings: public BaseLayer {
private:
    // params
    size_t pooled_projection_dim_;
    size_t embedding_dim_;
    size_t embedding_input_dim_ = 256;  // 写死在FLUX当中
    bool   need_silu_;

    // buffer
    T* timestep_proj_buffer = nullptr;
    T* guidance_proj_buffer = nullptr;
    T* timestep_emb_buffer  = nullptr;
    T* guidance_emb_buffer  = nullptr;
    // T* guidance_emb_buffer  = nullptr;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t outputBytes);

public:
    TimeProjection<T>* time_proj = nullptr;

    TimestepEmbeddingBlock<T>* timestep_emb = nullptr;
    TimestepEmbeddingBlock<T>* guidance_emb = nullptr;
    TimestepEmbeddingBlock<T>* text_temb    = nullptr;

    CombinedTimestepGuidanceTextProjEmbeddings(const size_t     pooled_projection_dim,
                                               const size_t     embedding_dim,
                                               const size_t     embedding_input_dim,
                                               const bool       need_silu,
                                               cudaStream_t     stream,
                                               cublasMMWrapper* cublas_wrapper,
                                               IAllocator*      allocator,
                                               const bool       is_free_buffer_after_forward,
                                               const bool       sparse);

    // CombinedTimestepGuidanceTextProjEmbeddings(CombinedTimestepGuidanceTextProjEmbeddings<T> const& other);

    virtual ~CombinedTimestepGuidanceTextProjEmbeddings();

    virtual void forward(TensorMap*                                                 output_tensors,
                         const TensorMap*                                           input_tensors,
                         const float                                                timestep,
                         const float                                                guidance,
                         const CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>* weight);
};

}  // namespace lyradiff
