#include "FluxAttnPostInt4Processor.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxAttnPostInt4Processor<T>::FluxAttnPostInt4Processor(size_t           embedding_dim,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        const bool       is_free_buffer_after_forward,
                                                        const bool       sparse):
    FluxAttnPostProcessor<T>(embedding_dim, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse)
{
    if (std::is_same<T, half>::value) {
        this->cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        this->cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        this->cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }

    geglu_gemm = new W4A4Gemm<T>(this->embedding_dim_ * 4,
                                 this->embedding_dim_,
                                 32,
                                 true,
                                 64,
                                 this->stream_,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 sparse);
}

template<typename T>
FluxAttnPostInt4Processor<T>::FluxAttnPostInt4Processor(FluxAttnPostInt4Processor<T> const& other):
    FluxAttnPostProcessor<T>(other.embedding_dim_,
                             other.stream_,
                             other.cublas_wrapper_,
                             other.allocator_,
                             other.is_free_buffer_after_forward_,
                             other.sparse_)
{
    if (std::is_same<T, half>::value) {
        this->cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        this->cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        this->cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
void FluxAttnPostInt4Processor<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxAttnPostInt4Processor::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxAttnPostInt4Processor<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t hidden_buffer_size = sizeof(T) * batch_size * seq_len * this->embedding_dim_;
    size_t ffn_buffer_size    = sizeof(T) * batch_size * seq_len * this->embedding_dim_ * 4;
    size_t fp8_buffer_size    = sizeof(__nv_fp8_e4m3) * batch_size * seq_len * this->embedding_dim_ * 4;

    hidden_buffer =
        (T*)this->allocator_->reMallocWithName("FluxAttnPostProcessor_hidden_buffer", hidden_buffer_size, false);
    hidden_buffer2 =
        (T*)this->allocator_->reMallocWithName("FluxAttnPostProcessor_hidden_buffer2", hidden_buffer_size, false);
    ffn_inner_buffer =
        (T*)this->allocator_->reMallocWithName("FluxAttnPostProcessor_ffn_inner_buffer", ffn_buffer_size, false);

    fp8_buffer = (__nv_fp8_e4m3*)this->allocator_->reMallocWithName("fp8_input_buffer", fp8_buffer_size, false);
}

template<typename T>
void FluxAttnPostInt4Processor<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void FluxAttnPostInt4Processor<T>::forward(const TensorMap*                          output_tensors,
                                           const TensorMap*                          input_tensors,
                                           const FluxAttnPostInt4ProcessorWeight<T>* weights)
{
    Tensor input_tensor  = input_tensors->at("input");
    Tensor attn_tensor   = input_tensors->at("attn_output");
    Tensor msa_tensor    = input_tensors->at("msa_input");
    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t seq_len    = input_tensor.shape[1];

    allocateBuffer(batch_size, seq_len);

    T* gate_msa_buffer  = &msa_tensor.getPtr<T>()[2 * batch_size * this->embedding_dim_];
    T* shift_mlp_buffer = &msa_tensor.getPtr<T>()[3 * batch_size * this->embedding_dim_];
    T* scale_mlp_buffer = &msa_tensor.getPtr<T>()[4 * batch_size * this->embedding_dim_];
    T* gate_mlp_buffer  = &msa_tensor.getPtr<T>()[5 * batch_size * this->embedding_dim_];

    invokeFusedGateAndResidual(hidden_buffer,
                               attn_tensor.getPtr<T>(),
                               gate_msa_buffer,
                               input_tensor.getPtr<T>(),
                               batch_size,
                               seq_len,
                               this->embedding_dim_,
                               this->stream_);

    Tensor hidden_buffer_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, hidden_buffer);
    // hidden_buffer_tensor.saveNpy("hidden_buffer_tensor.npy");

    invokeLayerNormWithShiftAndScale(hidden_buffer2,
                                     hidden_buffer,
                                     scale_mlp_buffer,
                                     shift_mlp_buffer,
                                     batch_size,
                                     seq_len,
                                     this->embedding_dim_,
                                     this->stream_,
                                     1e-6);

    Tensor geglu_input_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_}, hidden_buffer2);

    Tensor ffn_inner_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_ * 4}, ffn_inner_buffer);

    TensorMap input_map  = TensorMap({{"input", geglu_input_tensor}});
    TensorMap output_map = TensorMap({{"output", ffn_inner_tensor}});

    geglu_gemm->forward(&output_map, &input_map, weights->gelu_proj_weight);

    invokeFusedBiasAndGelu(ffn_inner_buffer,
                           ffn_inner_buffer,
                           weights->gelu_proj_weight->bias,
                           batch_size,
                           seq_len,
                           this->embedding_dim_ * 4,
                           this->stream_);

    // geglu_input_tensor.saveNpy("geglu_input_tensor.npy");

    invokeCudaD2DScaleCpyConvert(fp8_buffer,
                                 ffn_inner_buffer,
                                 weights->ff_linear_input_scale,
                                 true,
                                 batch_size * seq_len * this->embedding_dim_ * 4,
                                 this->stream_);

    int m_1 = batch_size * seq_len;
    int n_1 = this->embedding_dim_;
    int k_1 = this->embedding_dim_ * 4;

    this->cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   n_1,                              // m
                                                   m_1,                              // n
                                                   k_1,                              // k
                                                   weights->ff_linear_weight,        // A
                                                   k_1,                              // LDA
                                                   weights->ff_linear_weight_scale,  // weight scale
                                                   fp8_buffer,                       // B
                                                   k_1,                              // LDB
                                                   weights->ff_linear_input_scale,   // input scale
                                                   hidden_buffer2,                   // C
                                                   n_1,                              // LDC
                                                   weights->ff_linear_bias,          // bias
                                                   nullptr,                          // residual
                                                   1.0f,                             // alpha
                                                   0.0f);                            // beta

    Tensor ffn_res_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, hidden_buffer2);
    // ffn_res_tensor.saveNpy("ffn_res_tensor.npy");

    invokeFusedGateAndResidual(output_tensor.getPtr<T>(),
                               hidden_buffer2,
                               gate_mlp_buffer,
                               hidden_buffer,
                               batch_size,
                               seq_len,
                               this->embedding_dim_,
                               this->stream_);

    if (this->is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxAttnPostInt4Processor<T>::~FluxAttnPostInt4Processor()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class FluxAttnPostInt4Processor<float>;
template class FluxAttnPostInt4Processor<half>;
#ifdef ENABLE_BF16
template class FluxAttnPostInt4Processor<__nv_bfloat16>;
#endif
}  // namespace lyradiff