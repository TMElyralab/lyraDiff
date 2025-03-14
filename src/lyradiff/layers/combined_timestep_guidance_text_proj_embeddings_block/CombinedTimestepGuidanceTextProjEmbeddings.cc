#include "CombinedTimestepGuidanceTextProjEmbeddings.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/basic/gemm.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

// initialize
template<typename T>
CombinedTimestepGuidanceTextProjEmbeddings<T>::CombinedTimestepGuidanceTextProjEmbeddings(
    const size_t     pooled_projection_dim,
    const size_t     embedding_dim,
    const size_t     embedding_input_dim,
    const bool       need_silu,
    cudaStream_t     stream,
    cublasMMWrapper* cublas_wrapper,
    IAllocator*      allocator,
    const bool       is_free_buffer_after_forward,
    const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    pooled_projection_dim_(pooled_projection_dim),
    embedding_dim_(embedding_dim),
    embedding_input_dim_(embedding_input_dim),
    need_silu_(need_silu)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (typeid(T) == typeid(half)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (typeid(T) == typeid(float)) {
        // printf("set cublas_wrapper to fp32 mode\n");
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to bf16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif

    time_proj = new TimeProjection<T>(
        embedding_input_dim_, true, 0, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse);

    timestep_emb = new TimestepEmbeddingBlock<T>(embedding_input_dim_,
                                                 embedding_dim_,
                                                 embedding_dim_,
                                                 stream,
                                                 cublas_wrapper,
                                                 allocator,
                                                 is_free_buffer_after_forward,
                                                 false);

    guidance_emb = new TimestepEmbeddingBlock<T>(embedding_input_dim_,
                                                 embedding_dim_,
                                                 embedding_dim_,
                                                 stream,
                                                 cublas_wrapper,
                                                 allocator,
                                                 is_free_buffer_after_forward,
                                                 false);

    text_temb = new TimestepEmbeddingBlock<T>(pooled_projection_dim_,
                                              embedding_dim_,
                                              embedding_dim_,
                                              stream,
                                              cublas_wrapper,
                                              allocator,
                                              is_free_buffer_after_forward,
                                              false);
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddings<T>::forward(
    TensorMap*                                                 output_tensors,
    const TensorMap*                                           input_tensors,
    const float                                                timestep,
    const float                                                guidance,
    const CombinedTimestepGuidanceTextProjEmbeddingsWeight<T>* weight)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor pooled_projection_tensor = input_tensors->at("pooled_projection");
    Tensor output_tensor            = output_tensors->at("output");

    // input tensor [bs, input_dim]
    size_t bs = pooled_projection_tensor.shape[0];
    // size_t timeid_length   = pooled_projection_tensor.shape[1];
    // size_t text_emb_length = text_emb_tensor.shape[1];
    // printf("read test data: batch_size=%d, input_dim=%d\n", bs, input_dim);

    T*     output      = output_tensor.getPtr<T>();
    size_t outputBytes = sizeof(T) * output_tensor.size();

    allocateBuffer(bs);
    Tensor t_proj = Tensor(MEMORY_GPU, output_tensor.type, {bs, embedding_input_dim_}, timestep_proj_buffer);
    Tensor g_proj = Tensor(MEMORY_GPU, output_tensor.type, {bs, embedding_input_dim_}, guidance_proj_buffer);

    Tensor t_emb = Tensor(MEMORY_GPU, output_tensor.type, {bs, embedding_dim_}, timestep_emb_buffer);
    Tensor g_emb = Tensor(MEMORY_GPU, output_tensor.type, {bs, embedding_dim_}, guidance_emb_buffer);

    time_proj->forward(t_proj, timestep);
    time_proj->forward(g_proj, guidance);

    // t_proj.saveNpy("t_proj.npy");
    // g_proj.saveNpy("g_proj.npy");

    TensorMap input_map1({{"input", t_proj}});
    TensorMap output_map1({{"output", t_emb}});

    timestep_emb->forward(&output_map1, &input_map1, weight->timestep_emb_weight);

    // t_emb.saveNpy("t_emb.npy");

    TensorMap input_map2({{"input", g_proj}});
    TensorMap output_map2({{"output", g_emb}});

    guidance_emb->forward(&output_map2, &input_map2, weight->guidance_emb_weight);

    // g_emb.saveNpy("g_emb.npy");

    TensorMap input_map3({{"input", pooled_projection_tensor}});
    TensorMap output_map3({{"output", output_tensor}});

    text_temb->forward(&output_map3, &input_map3, weight->text_emb_weight);
    // output_tensor.saveNpy("pooled_emb.npy");

    // add two tensor together.
    invokeAdd3Tensor2d(output_tensor.getPtr<T>(),
                       output_tensor.getPtr<T>(),
                       guidance_emb_buffer,
                       timestep_emb_buffer,
                       bs,
                       embedding_dim_,
                       need_silu_,
                       getStream());

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
CombinedTimestepGuidanceTextProjEmbeddings<T>::~CombinedTimestepGuidanceTextProjEmbeddings()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddings<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "TimestepEmbeddingBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddings<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t proj_buffer_size      = sizeof(T) * batch_size * embedding_input_dim_;
    size_t embedding_buffer_size = sizeof(T) * batch_size * embedding_dim_;
    // size_t add_emb_cat_buffer_size = sizeof(T) * batch_size * augemb_temb_input_dim_;
    // size_t output_buffer_size      = sizeof(T) * batch_size * temb_channels_;

    timestep_proj_buffer =
        (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_timestep_proj_buffer", proj_buffer_size, false);
    guidance_proj_buffer =
        (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_guidance_proj_buffer", proj_buffer_size, false);
    timestep_emb_buffer =
        (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_timestep_emb_buffer", embedding_buffer_size, false);
    guidance_emb_buffer =
        (T*)allocator_->reMallocWithName("TextTimeEmbeddingBlock_guidance_emb_buffer", embedding_buffer_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void CombinedTimestepGuidanceTextProjEmbeddings<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&timestep_proj_buffer));
        allocator_->free((void**)(&guidance_proj_buffer));
        allocator_->free((void**)(&timestep_emb_buffer));
        allocator_->free((void**)(&guidance_emb_buffer));
        is_allocate_buffer_ = false;
    }
}

template class CombinedTimestepGuidanceTextProjEmbeddings<half>;
template class CombinedTimestepGuidanceTextProjEmbeddings<float>;
#ifdef ENABLE_BF16
template class CombinedTimestepGuidanceTextProjEmbeddings<__nv_bfloat16>;
#endif

}  // namespace lyradiff