#include "FluxAttnPostProcessor.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxAttnPostProcessor<T>::FluxAttnPostProcessor(size_t           embedding_dim,
                                                cudaStream_t     stream,
                                                cublasMMWrapper* cublas_wrapper,
                                                IAllocator*      allocator,
                                                const bool       is_free_buffer_after_forward,
                                                const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    embedding_dim_(embedding_dim)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
FluxAttnPostProcessor<T>::FluxAttnPostProcessor(FluxAttnPostProcessor<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    embedding_dim_(other.embedding_dim_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
void FluxAttnPostProcessor<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxAttnPostProcessor::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxAttnPostProcessor<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t hidden_buffer_size = sizeof(T) * batch_size * seq_len * embedding_dim_;
    size_t ffn_buffer_size    = sizeof(T) * batch_size * seq_len * embedding_dim_ * 4;

    hidden_buffer = (T*)allocator_->reMallocWithName("FluxAttnPostProcessor_hidden_buffer", hidden_buffer_size, false);
    hidden_buffer2 =
        (T*)allocator_->reMallocWithName("FluxAttnPostProcessor_hidden_buffer2", hidden_buffer_size, false);
    ffn_inner_buffer =
        (T*)allocator_->reMallocWithName("FluxAttnPostProcessor_ffn_inner_buffer", ffn_buffer_size, false);
}

template<typename T>
void FluxAttnPostProcessor<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void FluxAttnPostProcessor<T>::forward(const TensorMap*                      output_tensors,
                                       const TensorMap*                      input_tensors,
                                       const FluxAttnPostProcessorWeight<T>* weights)
{
    Tensor input_tensor  = input_tensors->at("input");
    Tensor attn_tensor   = input_tensors->at("attn_output");
    Tensor msa_tensor    = input_tensors->at("msa_input");
    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t seq_len    = input_tensor.shape[1];

    allocateBuffer(batch_size, seq_len);

    T* gate_msa_buffer  = &msa_tensor.getPtr<T>()[2 * batch_size * embedding_dim_];
    T* shift_mlp_buffer = &msa_tensor.getPtr<T>()[3 * batch_size * embedding_dim_];
    T* scale_mlp_buffer = &msa_tensor.getPtr<T>()[4 * batch_size * embedding_dim_];
    T* gate_mlp_buffer  = &msa_tensor.getPtr<T>()[5 * batch_size * embedding_dim_];

    invokeFusedGateAndResidual(hidden_buffer,
                               attn_tensor.getPtr<T>(),
                               gate_msa_buffer,
                               input_tensor.getPtr<T>(),
                               batch_size,
                               seq_len,
                               embedding_dim_,
                               stream_);

    // Tensor hidden_buffer_tensor =
    //     Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, hidden_buffer);
    // hidden_buffer_tensor.saveNpy("hidden_buffer_tensor.npy");

    invokeLayerNormWithShiftAndScale(hidden_buffer2,
                                     hidden_buffer,
                                     scale_mlp_buffer,
                                     shift_mlp_buffer,
                                     batch_size,
                                     seq_len,
                                     embedding_dim_,
                                     stream_,
                                     1e-6);

    // Tensor norm_res_tensor =
    //     Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, hidden_buffer2);
    // norm_res_tensor.saveNpy("norm_res_tensor.npy");

    int m_1 = batch_size * seq_len;
    int n_1 = embedding_dim_ * 4;
    int k_1 = embedding_dim_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                        // m
                                             m_1,                        // n
                                             k_1,                        // k
                                             weights->gelu_proj_weight,  // A
                                             k_1,                        // LDA
                                             hidden_buffer2,             // B
                                             k_1,                        // LDB
                                             ffn_inner_buffer,           // C
                                             n_1,                        // LDC
                                             weights->gelu_proj_bias,    // bias
                                             nullptr,                    // residual
                                             1.0f,                       // alpha
                                             0.0f,                       // beta
                                             true);                      // gelu

    m_1 = batch_size * seq_len;
    n_1 = embedding_dim_;
    k_1 = embedding_dim_ * 4;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                        // m
                                             m_1,                        // n
                                             k_1,                        // k
                                             weights->ff_linear_weight,  // A
                                             k_1,                        // LDA
                                             ffn_inner_buffer,           // B
                                             k_1,                        // LDB
                                             hidden_buffer2,             // C
                                             n_1,                        // LDC
                                             weights->ff_linear_bias,    // bias
                                             nullptr,                    // residual
                                             1.0f,                       // alpha
                                             0.0f);                      // beta

    // Tensor ffn_res_tensor =
    //     Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, hidden_buffer2);
    // ffn_res_tensor.saveNpy("ffn_res_tensor.npy");

    invokeFusedGateAndResidual(output_tensor.getPtr<T>(),
                               hidden_buffer2,
                               gate_mlp_buffer,
                               hidden_buffer,
                               batch_size,
                               seq_len,
                               embedding_dim_,
                               stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxAttnPostProcessor<T>::~FluxAttnPostProcessor()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class FluxAttnPostProcessor<float>;
template class FluxAttnPostProcessor<half>;
#ifdef ENABLE_BF16
template class FluxAttnPostProcessor<__nv_bfloat16>;
#endif
}  // namespace lyradiff