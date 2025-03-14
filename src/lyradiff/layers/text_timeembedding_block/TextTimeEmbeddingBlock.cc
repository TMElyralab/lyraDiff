#include "src/lyradiff/layers/text_timeembedding_block/TextTimeEmbeddingBlock.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/basic/gemm.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

// initialize
template<typename T>
TextTimeEmbeddingBlock<T>::TextTimeEmbeddingBlock(const size_t     time_proj_out_dim,
                                                  const size_t     augemb_time_proj_out_dim,
                                                  const size_t     text_emb_dim,
                                                  const size_t     temb_channels,
                                                  cudaStream_t     stream,
                                                  cublasMMWrapper* cublas_wrapper,
                                                  IAllocator*      allocator,
                                                  const bool       is_free_buffer_after_forward,
                                                  const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    time_proj_out_dim_(time_proj_out_dim),
    augemb_time_proj_out_dim_(augemb_time_proj_out_dim),
    text_emb_dim_(text_emb_dim),
    temb_channels_(temb_channels)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (typeid(T) == typeid(half)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (typeid(T) == typeid(float)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP32GemmConfig();
    }

    augemb_time_proj = new TimeProjection<T>(augemb_time_proj_out_dim_,
                                      true,
                                      0,
                                      stream,
                                      cublas_wrapper,
                                      allocator,
                                      is_free_buffer_after_forward,
                                      sparse);
    augemb_temb_input_dim_ = augemb_time_proj_out_dim_*6 + text_emb_dim_;

    augemb_temb = new TimestepEmbeddingBlock<T>(augemb_temb_input_dim_,
                                                       temb_channels_,
                                                       temb_channels_,
                                                       stream,
                                                       cublas_wrapper,
                                                       allocator,
                                                       is_free_buffer_after_forward,
                                                       false);

    time_proj = new TimeProjection<T>(time_proj_out_dim_,
                                      true,
                                      0,
                                      stream,
                                      cublas_wrapper,
                                      allocator,
                                      is_free_buffer_after_forward,
                                      sparse);

    time_emb = new TimestepEmbeddingBlock<T>(time_proj_out_dim_,
                                             temb_channels_,
                                             temb_channels_,
                                             stream,
                                             cublas_wrapper,
                                             allocator,
                                             is_free_buffer_after_forward,
                                             false);

}

template<typename T>
void TextTimeEmbeddingBlock<T>::forward(TensorMap*                             output_tensors,
                                        const TensorMap*                       input_tensors,
                                        const float                                 timestep,
                                        const TextTimeEmbeddingBlockWeight<T>* text_timeembedding_block_weight)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor text_emb_tensor = input_tensors->at("text_emb");
    Tensor time_id_tensor = input_tensors->at("time_id");
    Tensor output_tensor = output_tensors->at("output");

    // input tensor [bs, input_dim]
    size_t bs = time_id_tensor.shape[0];
    size_t timeid_length = time_id_tensor.shape[1];
    size_t text_emb_length = text_emb_tensor.shape[1];
    // printf("read test data: batch_size=%d, input_dim=%d\n", bs, input_dim);

    T*     output            = output_tensor.getPtr<T>();
    size_t outputBytes = sizeof(T) * output_tensor.size();

    allocateBuffer(bs);
    Tensor temb = Tensor(MEMORY_GPU, output_tensor.type, {bs, time_proj_out_dim_}, temb_buffer_);
    Tensor aug_temb_input = Tensor(MEMORY_GPU, output_tensor.type, {bs, augemb_time_proj_out_dim_*timeid_length}, augemb_temb_buffer_);
    Tensor inner_output_tensor = Tensor(MEMORY_GPU, output_tensor.type, output_tensor.shape, output_buffer_);

    time_proj->forward(temb, timestep);
    augemb_time_proj->forward(aug_temb_input, time_id_tensor);
    // cat by last axis, add_temb shape: (2,1536)
    invokeConcat2d(augemb_cat_tproj_text_buf_, text_emb_tensor.getPtr<T>(), aug_temb_input.getPtr<T>(), bs, text_emb_length, augemb_time_proj_out_dim_*timeid_length, getStream());
    time_emb->forward(output_tensor, temb, text_timeembedding_block_weight->timestep_emb_weight);

    Tensor aug_emb_cat_out = Tensor(MEMORY_GPU, output_tensor.type, {bs,augemb_temb_input_dim_}, augemb_cat_tproj_text_buf_);
    augemb_temb->forward(inner_output_tensor, aug_emb_cat_out, text_timeembedding_block_weight->augemb_weight);

    // add two tensor together.
    invokeAddTensor2d(output_tensor.getPtr<T>(), output_tensor.getPtr<T>(), inner_output_tensor.getPtr<T>(), bs, temb_channels_, getStream());

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
TextTimeEmbeddingBlock<T>::~TextTimeEmbeddingBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void TextTimeEmbeddingBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "TimestepEmbeddingBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void TextTimeEmbeddingBlock<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t temb_buffer_size = sizeof(T)*batch_size*time_proj_out_dim_;
    size_t augemb_temb_buffer_size = sizeof(T)*batch_size*augemb_time_proj_out_dim_*6;
    size_t add_emb_cat_buffer_size = sizeof(T)*batch_size*augemb_temb_input_dim_;
    size_t output_buffer_size = sizeof(T)*batch_size*temb_channels_;
    
    temb_buffer_ = (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_temb_buffer_", temb_buffer_size, false);
    augemb_temb_buffer_ = (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_augemb_temb_buffer_", augemb_temb_buffer_size, false);
    output_buffer_ =  (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_output_buffer_", output_buffer_size, false);
    augemb_cat_tproj_text_buf_ = (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_augemb_cat_tproj_text_buf_", add_emb_cat_buffer_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void TextTimeEmbeddingBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&temb_buffer_));
        allocator_->free((void**)(&augemb_temb_buffer_));
        allocator_->free((void**)(&output_buffer_));
        allocator_->free((void**)(&augemb_cat_tproj_text_buf_));
        is_allocate_buffer_ = false;
    }
}

template class TextTimeEmbeddingBlock<half>;
template class TextTimeEmbeddingBlock<float>;

}  // namespace lyradiff