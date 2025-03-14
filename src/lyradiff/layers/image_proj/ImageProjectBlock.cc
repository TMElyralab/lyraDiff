#include "src/lyradiff/layers/image_proj/ImageProjectBlock.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/layers/image_proj/ImageProjectWeight.h"

using namespace std;

namespace lyradiff {
template<typename T>
ImageProjection<T>::ImageProjection(const size_t     image_embed_dim_,
                                    const size_t     cross_attention_dim_,
                                    const size_t     num_image_text_embeds_,
                                    cudaStream_t     stream,
                                    cublasMMWrapper* cublas_wrapper,
                                    IAllocator*      allocator,
                                    const bool       is_free_buffer_after_forward,
                                    const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    image_embed_dim_(image_embed_dim_),
    cross_attention_dim_(cross_attention_dim_),
    num_image_text_embeds_(num_image_text_embeds_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (typeid(T) == typeid(half)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP16GemmConfig();
    }
    if (typeid(T) == typeid(float)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP32GemmConfig();
    }
}

template<typename T>
ImageProjection<T>::~ImageProjection()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void ImageProjection<T>::forward(TensorMap*                        output_tensors,
                                 TensorMap*                        input_tensors,
                                 const ImageProjectBlockWeight<T>* image_proj_weight)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor input_tensor  = input_tensors->at("input");
    Tensor output_tensor = output_tensors->at("output");

    forward(output_tensor, input_tensor, image_proj_weight);
}

template<typename T>
void ImageProjection<T>::forward(Tensor&                           output_tensor,
                                 Tensor&                           input_tensor,
                                 const ImageProjectBlockWeight<T>* image_proj_weight)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // input tensor [bs, input_dim]
    int bs = input_tensor.shape[0];

    T* input  = input_tensor.getPtr<T>();
    T* output = output_tensor.getPtr<T>();

    size_t inputTensorBytes  = sizeof(T) * input_tensor.size();
    size_t outputTensorBytes = sizeof(T) * output_tensor.size();

    if(bs > pre_batch_size_) {
        allocateBuffer(bs);
        pre_batch_size_ = bs;
    }

    int m_0 = bs;
    int n_0 = num_image_text_embeds_ * cross_attention_dim_;
    int k_0 = image_embed_dim_;

    // mxk kxn -> mxn
    // [2, 1024] [1024, 3072]

    // 2 3072 1024
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n_0,                               // m
                          m_0,                               // n
                          k_0,                               // k
                          image_proj_weight->linear_weight,  // A
                          k_0,                               // LDA
                          input,                             // B
                          k_0,                               // LDB
                          inter_buf_0_0_,                    // C
                          n_0);

    // reshape to [bs, num_image_text_embeds, -1]
    size_t seq_len = 1;
    size_t dim_    = num_image_text_embeds_ * cross_attention_dim_;
    invokeaddBiasToGEMM<T>(inter_buf_0_0_, image_proj_weight->linear_bias, bs, seq_len, dim_, getStream());
    // 3072
    invokeLayerNorm<T>(output,
                       inter_buf_0_0_,
                       image_proj_weight->norm_gamma,
                       image_proj_weight->norm_beta,
                       bs,
                       num_image_text_embeds_,
                       cross_attention_dim_,
                       getStream());
}

template<typename T>
void ImageProjection<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "ImageProjection::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void ImageProjection<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t inter_buf_0_0_size_ = sizeof(T) * batch_size * num_image_text_embeds_ * cross_attention_dim_;
    size_t inter_buf_0_1_size_ = sizeof(T) * batch_size * num_image_text_embeds_ * cross_attention_dim_;

    inter_buf_0_0_ = (T*)allocator_->reMallocWithName("ImageProjection_inter_buf_0_0_", inter_buf_0_0_size_, false);
    inter_buf_0_1_ = (T*)allocator_->reMallocWithName("ImageProjection_inter_buf_0_1_", inter_buf_0_1_size_, false);
}

template<typename T>
void ImageProjection<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_0_0_));
        allocator_->free((void**)(&inter_buf_0_1_));
        is_allocate_buffer_ = false;
    }
}

template class ImageProjection<float>;
template class ImageProjection<half>;
}  // namespace lyradiff
