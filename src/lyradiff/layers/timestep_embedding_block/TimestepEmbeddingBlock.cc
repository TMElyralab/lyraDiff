#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlock.h"
#include "src/lyradiff/kernels/basic/gemm.h"
#include "src/lyradiff/kernels/timestep_embedding/ffn_kernels.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

// initialize
template<typename T>
TimestepEmbeddingBlock<T>::TimestepEmbeddingBlock(const size_t     input_dim_,
                                                  const size_t     output_dim_0_,
                                                  const size_t     output_dim_,
                                                  cudaStream_t     stream,
                                                  cublasMMWrapper* cublas_wrapper,
                                                  IAllocator*      allocator,
                                                  const bool       is_free_buffer_after_forward,
                                                  const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    input_dim_(input_dim_),
    output_dim_0_(output_dim_0_),
    output_dim_(output_dim_)
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
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
}

template<typename T>
void TimestepEmbeddingBlock<T>::forward(Tensor&                                output_tensor,
                                        const Tensor&                          input_tensor,
                                        const TimestepEmbeddingBlockWeight<T>* timestep_embedding_block_weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // input tensor [bs, input_dim]
    int bs = input_tensor.shape[0];
    // printf("read test data: batch_size=%d, input_dim=%d\n", bs, input_dim);

    T*     input             = input_tensor.getPtr<T>();
    T*     output            = output_tensor.getPtr<T>();
    size_t inputTensorBytes  = sizeof(T) * input_tensor.size();
    size_t outputTensorBytes = sizeof(T) * output_tensor.size();

    // printf("ready to infer\n");
    allocateBuffer(bs);

    // Step 0:
    // input_tensor -> linear layer -> activation layer(SiLu)
    // [bs, input_dim] -> [bs, output_dim_0_] -> [bs, output_dim_0_]
    int m_0 = bs;
    int n_0 = output_dim_0_;
    int k_0 = input_dim_;

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n_0,                                               // m
                          m_0,                                               // n
                          k_0,                                               // k
                          timestep_embedding_block_weights->linear1_weight,  // A
                          k_0,                                               // LDA
                          input,                                             // B
                          k_0,                                               // LDB
                          inter_buf_0_0_,                                    // C
                          n_0                                                // LDC
    );

    // Tensor inter_buf_0_0_tensor_ = Tensor(MEMORY_GPU, TYPE_FP32, {bs, output_dim_0_}, inter_buf_0_0_);
    // inter_buf_0_0_tensor_.saveNpy(
    //     "/group/30063/users/carsonhxsu/FasterStableDiffusion/tests/timestep_embedding_test/data/cpp_linear_1_weight_output.npy");

    invokeFusedAddBiasSilu1D(
        inter_buf_0_1_, inter_buf_0_0_, timestep_embedding_block_weights->linear1_bias, output_dim_0_, bs, getStream());

    // Step 1:
    // inter_buf_0_1_ -> linear layer
    // [bs, output_dim_0_] -> [bs, output_dim_]
    int m_1 = bs;
    int n_1 = output_dim_;
    int k_1 = output_dim_0_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                                               // m
                                             m_1,                                               // n
                                             k_1,                                               // k
                                             timestep_embedding_block_weights->linear2_weight,  // A
                                             k_1,                                               // LDA
                                             inter_buf_0_1_,                                    // B
                                             k_1,                                               // LDB
                                             output,                                            // C
                                             n_1,                                               // LDC
                                             timestep_embedding_block_weights->linear2_bias,    // bias
                                             nullptr,                                           // residual
                                             1.0f,                                              // alpha
                                             0.0f);                                             // beta

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void TimestepEmbeddingBlock<T>::forward(TensorMap*                             output_tensors,
                                        const TensorMap*                       input_tensors,
                                        const TimestepEmbeddingBlockWeight<T>* timestep_embedding_block_weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor input_tensor  = input_tensors->at("input");
    Tensor output_tensor = output_tensors->at("output");
    forward(output_tensor, input_tensor, timestep_embedding_block_weights);
}

template<typename T>
TimestepEmbeddingBlock<T>::~TimestepEmbeddingBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void TimestepEmbeddingBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "TimestepEmbeddingBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void TimestepEmbeddingBlock<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // linear 1 + activation (silu)
    size_t inter_buf_0_0_size_ = sizeof(T) * batch_size * output_dim_0_;
    size_t inter_buf_0_1_size_ = sizeof(T) * batch_size * output_dim_0_;

    inter_buf_0_0_ =
        (T*)allocator_->reMallocWithName("TimestepEmbeddingBlock_inter_buf_0_0_", inter_buf_0_0_size_, false);
    inter_buf_0_1_ =
        (T*)allocator_->reMallocWithName("TimestepEmbeddingBlock_inter_buf_0_1_", inter_buf_0_1_size_, false);

    // linear 2
    size_t inter_buf_1_0_size_ = sizeof(T) * batch_size * output_dim_;
    size_t inter_buf_1_1_size_ = sizeof(T) * batch_size * output_dim_;

    inter_buf_1_0_ =
        (T*)allocator_->reMallocWithName("TimestepEmbeddingBlock_inter_buf_1_0_", inter_buf_1_0_size_, false);
    inter_buf_1_1_ =
        (T*)allocator_->reMallocWithName("TimestepEmbeddingBlock_inter_buf_1_1_", inter_buf_1_1_size_, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void TimestepEmbeddingBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_0_0_));
        allocator_->free((void**)(&inter_buf_0_1_));

        allocator_->free((void**)(&inter_buf_1_0_));
        allocator_->free((void**)(&inter_buf_1_1_));

        is_allocate_buffer_ = false;
    }
}

template class TimestepEmbeddingBlock<half>;
template class TimestepEmbeddingBlock<float>;

#ifdef ENABLE_BF16
template class TimestepEmbeddingBlock<__nv_bfloat16>;
#endif

}  // namespace lyradiff