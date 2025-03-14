#include "AdaLayerNorm.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"

using namespace std;
namespace lyradiff {

template<typename T>
AdaLayerNorm<T>::AdaLayerNorm(size_t           embedding_dim,
                              size_t           embedding_scale,
                              bool             switch_scale,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              const bool       is_free_buffer_after_forward,
                              const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    embedding_dim_(embedding_dim),
    embedding_scale_(embedding_scale),
    switch_scale_(switch_scale)
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
AdaLayerNorm<T>::AdaLayerNorm(AdaLayerNorm<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    embedding_dim_(other.embedding_dim_),
    embedding_scale_(other.embedding_scale_)
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
void AdaLayerNorm<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "AdaLayerNorm::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void AdaLayerNorm<T>::allocateBuffer(size_t batch_size)
{
    size_t msa_buffer_size = sizeof(T) * batch_size * embedding_dim_ * embedding_scale_;
    msa_buffer             = (T*)allocator_->reMallocWithName("AdaLayerNorm_msa_buffer", msa_buffer_size, false);
}

template<typename T>
void AdaLayerNorm<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

// 所有Flux层的AdaLayerNorm都复用这个Layer，通过 embedding_scale 这个数值判断不同ada layernorm 需要返回的数据量
template<typename T>
void AdaLayerNorm<T>::forward(const TensorMap*             output_tensors,
                              const TensorMap*             input_tensors,
                              const AdaLayerNormWeight<T>* weights)
{
    // 这里默认temb 已经被silu过了，因为flux里面所有ada layernorm 层都有对temb的前置silu，这里放到外面做了
    // input_tensor       -> [B, S, N * D]
    // temb_tensor        -> [B, N * D]
    // output_tensor      -> [B, S, N * D]
    // msa_output_tensor  -> [embedding_scale, B, N * D]
    Tensor input_tensor      = input_tensors->at("input");
    Tensor temb_tensor       = input_tensors->at("temb");
    Tensor output_tensor     = output_tensors->at("output");
    Tensor msa_output_tensor = output_tensors->at("msa_output");

    size_t batch_size   = input_tensor.shape[0];
    size_t seq_len      = input_tensor.shape[1];
    size_t ret_msa_size = msa_output_tensor.shape[0];

    if (ret_msa_size != embedding_scale_) {
        cout << "ret_msa_size and embedding_scale_ not equal" << endl;
        throw "ret_msa_size and embedding_scale_ not equal";
    }

    // cout << "input_tensor: " << input_tensor.toString() << endl;
    // cout << "temb_tensor: " << temb_tensor.toString() << endl;
    // cout << "output_tensor: " << output_tensor.toString() << endl;
    // cout << "msa_output_tensor: " << msa_output_tensor.toString() << endl;

    int m_1 = batch_size;
    int n_1 = embedding_dim_ * embedding_scale_;
    int k_1 = embedding_dim_;

    // temb_tensor.saveNpy("temb_tensor1.npy");

    // cout << "ada layernorm m n k: " << m_1 << " " << n_1 << " " << k_1 << endl;
    if (batch_size > 1) {
        allocateBuffer(batch_size);
        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n_1,                      // m
                                                 m_1,                      // n
                                                 k_1,                      // k
                                                 weights->linear_weight,   // A
                                                 k_1,                      // LDA
                                                 temb_tensor.getPtr<T>(),  // B
                                                 k_1,                      // LDB
                                                 msa_buffer,               // C
                                                 n_1,                      // LDC
                                                 weights->linear_bias,     // bias
                                                 nullptr,                  // residual
                                                 1.0f,                     // alpha
                                                 0.0f);                    // beta

        invokeTranspose102(msa_output_tensor.getPtr<T>(), msa_buffer, batch_size, embedding_scale_, embedding_dim_);
    }
    else {
        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n_1,                            // m
                                                 m_1,                            // n
                                                 k_1,                            // k
                                                 weights->linear_weight,         // A
                                                 k_1,                            // LDA
                                                 temb_tensor.getPtr<T>(),        // B
                                                 k_1,                            // LDB
                                                 msa_output_tensor.getPtr<T>(),  // C
                                                 n_1,                            // LDC
                                                 weights->linear_bias,           // bias
                                                 nullptr,                        // residual
                                                 1.0f,                           // alpha
                                                 0.0f);                          // beta
    }
    // temb_tensor.saveNpy("temb_tensor2.npy");
    // msa_output_tensor.saveNpy("msa_output_tensor.npy");

    T* scale_msa_ptr = &msa_output_tensor.getPtr<T>()[batch_size * embedding_dim_];
    T* shift_msa_ptr = msa_output_tensor.getPtr<T>();

    if (switch_scale_) {
        scale_msa_ptr = msa_output_tensor.getPtr<T>();
        shift_msa_ptr = &msa_output_tensor.getPtr<T>()[batch_size * embedding_dim_];
    }

    // msa_output_tensor.saveNpy("msa_output_tensor1.npy");

    invokeLayerNormWithShiftAndScale(output_tensor.getPtr<T>(),
                                     input_tensor.getPtr<T>(),
                                     scale_msa_ptr,
                                     shift_msa_ptr,
                                     batch_size,
                                     seq_len,
                                     embedding_dim_,
                                     stream_,
                                     1e-6);

    // msa_output_tensor.saveNpy("msa_output_tensor2.npy");

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
AdaLayerNorm<T>::~AdaLayerNorm()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class AdaLayerNorm<float>;
template class AdaLayerNorm<half>;
#ifdef ENABLE_BF16
template class AdaLayerNorm<__nv_bfloat16>;
#endif
}  // namespace lyradiff