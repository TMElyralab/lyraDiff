#include "ZeroCrossAttn.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/mid_block_2d/add_bias.h"

using namespace std;
namespace lyradiff {

template<typename T>
ZeroCrossAttn<T>::ZeroCrossAttn(size_t           query_dim,
                                size_t           context_dim,
                                cudaStream_t     stream,
                                cublasMMWrapper* cublas_wrapper,
                                IAllocator*      allocator,
                                const bool       is_free_buffer_after_forward,
                                const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    query_dim_(query_dim),
    context_dim_(context_dim)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }

    heads_     = query_dim_ / dim_head_;
    inner_dim_ = heads_ * dim_head_;

    int sm            = getSMVersion();
    flash_attn2_layer = new flash_attn2::FlashAttention2Layer<T>(
        heads_, dim_head_, sm, getStream(), cublas_wrapper_, allocator_, is_free_buffer_after_forward);
}

template<typename T>
ZeroCrossAttn<T>::ZeroCrossAttn(ZeroCrossAttn<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    query_dim_(other.query_dim_),
    context_dim_(other.context_dim_),
    heads_(other.heads_),
    inner_dim_(other.inner_dim_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
void ZeroCrossAttn<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "ZeroCrossAttn::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void ZeroCrossAttn<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t context_seq_len)
{
    size_t hidden_state_size = sizeof(T) * batch_size * seq_len * query_dim_;
    size_t context_size      = sizeof(T) * batch_size * context_seq_len * context_dim_;

    size_t attn_q_size     = sizeof(T) * batch_size * seq_len * inner_dim_;
    size_t attn_kv_size    = sizeof(T) * batch_size * context_seq_len * 2 * inner_dim_;
    size_t norm_cache_size = sizeof(double) * batch_size * norm_num_groups_ * 2;

    hidden_state_buf_ = (T*)allocator_->reMallocWithName("ZeroCrossAttn_hidden_state_buf_", hidden_state_size, false);
    context_buf_      = (T*)allocator_->reMallocWithName("ZeroCrossAttn_context_buf_", context_size, false);

    attn_q_buf_   = (T*)allocator_->reMallocWithName("ZeroCrossAttn_attn_q_buf_", attn_q_size, false);
    attn_kv_buf_  = (T*)allocator_->reMallocWithName("ZeroCrossAttn_attn_kv_buf_", attn_kv_size, false);
    attn_kv_buf2_ = (T*)allocator_->reMallocWithName("ZeroCrossAttn_attn_kv_buf2_", attn_kv_size, false);

    norm_cache_buf_ = (double*)allocator_->reMallocWithName("ZeroCrossAttn_norm_cache_buf_", norm_cache_size, false);

    attn_output_buf_ = (T*)allocator_->reMallocWithName("ZeroCrossAttn_attn_output_buf_", hidden_state_size, false);
}

template<typename T>
void ZeroCrossAttn<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void ZeroCrossAttn<T>::forward(TensorMap*                    output_tensors,
                               TensorMap*                    input_tensors,
                               const ZeroCrossAttnWeight<T>* weights,
                               const float                   control_scale)
{
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor context            = input_tensors->at("context");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];

    size_t seq_len         = init_hidden_states.shape[1] * init_hidden_states.shape[2];
    size_t context_seq_len = context.shape[1] * context.shape[2];

    allocateBuffer(batch_size, seq_len, context_seq_len);

    invokeGroupNorm<T>(hidden_state_buf_,
                       init_hidden_states.getPtr<T>(),
                       weights->norm1_gamma,
                       weights->norm1_beta,
                       norm_cache_buf_,
                       batch_size,
                       1,
                       seq_len,
                       query_dim_,
                       norm_num_groups_,
                       false,
                       getStream());

    invokeGroupNorm<T>(context_buf_,
                       context.getPtr<T>(),
                       weights->norm2_gamma,
                       weights->norm2_beta,
                       norm_cache_buf_,
                       batch_size,
                       1,
                       context_seq_len,
                       context_dim_,
                       norm_num_groups_,
                       false,
                       getStream());

    Tensor hidden_state_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, init_hidden_states.shape, hidden_state_buf_);
    Tensor context_tensor = Tensor(MEMORY_GPU, init_hidden_states.type, context.shape, context_buf_);

    // 计算 attn 的 q 和 kv
    int m = seq_len * batch_size;
    int n = inner_dim_;
    int k = inner_dim_;

    // cout << "计算 cross attention 的 q" << endl;
    // cout << "m: " << m << " n: " << n << " k: " << k << endl;

    cublas_wrapper_->Gemm(
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, weights->attention_q_weight, k, hidden_state_buf_, k, attn_q_buf_, n);

    m = context_seq_len * batch_size;
    n = inner_dim_ * 2;
    k = context_dim_;

    cublas_wrapper_->Gemm(
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, weights->attention_kv_weight, k, context_buf_, k, attn_kv_buf_, n);

    invokeCrossAttn2KernelInputPermute<T>(
        attn_kv_buf2_, attn_kv_buf_, batch_size, context_seq_len, inner_dim_, getStream());

    // kv_seq 太长。。需要flashattn2
    Tensor q_buf2 = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, heads_, dim_head_}, attn_q_buf_);
    Tensor k_buf2 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, context_seq_len, heads_, dim_head_}, attn_kv_buf2_);
    Tensor v_buf2 = Tensor(MEMORY_GPU,
                           init_hidden_states.type,
                           {batch_size, context_seq_len, heads_, dim_head_},
                           &attn_kv_buf2_[k_buf2.size()]);

    Tensor output_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, heads_, dim_head_}, attn_output_buf_);

    TensorMap input_map({{"q_buf", q_buf2}, {"k_buf", k_buf2}, {"v_buf", v_buf2}});
    TensorMap output_map({{"attn_output", output_tensor}});
    flash_attn2_layer->forward(&output_map, &input_map);

    // cross_attention_layer->forward(&cross_attn_output, &cross_attn_input);

    m = seq_len * batch_size;
    n = inner_dim_;
    k = inner_dim_;
    // 计算 attention to_out
    // cout << "计算 attention to_out " << endl;
    // cout << "m: " << m << " n: " << n << " k: " << k << endl;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n,
                                             m,
                                             k,
                                             weights->attention_to_out_weight,
                                             k,
                                             attn_output_buf_,
                                             k,
                                             output.getPtr<T>(),
                                             n,
                                             weights->attention_to_out_bias,
                                             init_hidden_states.getPtr<T>(),  // residual
                                             1.0f,                            // alpha
                                             1.0f);                           // beta

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
ZeroCrossAttn<T>::~ZeroCrossAttn()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class ZeroCrossAttn<float>;
template class ZeroCrossAttn<half>;
}  // namespace lyradiff