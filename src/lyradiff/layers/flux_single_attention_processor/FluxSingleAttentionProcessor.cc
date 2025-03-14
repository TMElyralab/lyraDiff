#include "FluxSingleAttentionProcessor.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxSingleAttentionProcessor<T>::FluxSingleAttentionProcessor(size_t           embedding_dim,
                                                              size_t           embedding_head_num,
                                                              size_t           embedding_head_dim,
                                                              cudaStream_t     stream,
                                                              cublasMMWrapper* cublas_wrapper,
                                                              IAllocator*      allocator,
                                                              const bool       is_free_buffer_after_forward,
                                                              const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    embedding_dim_(embedding_dim),
    embedding_head_num_(embedding_head_num),
    embedding_head_dim_(embedding_head_dim)
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

    float      q_scaling      = 1.0;
    const bool force_fp32_acc = false;
    const bool is_s_padded    = false;
    const bool causal_mask    = false;
    const bool has_alibi      = false;
    const bool scal_alibi     = false;
    const int  tp_size        = 1;
    const int  tp_rank        = 0;

    // printf("q_scaling: %f\n", q_scaling);
    int sm = getSMVersion();

    flash_attn_layer = new lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>(embedding_head_num_,
                                                                             embedding_head_dim_,
                                                                             embedding_head_num_,
                                                                             q_scaling,
                                                                             force_fp32_acc,
                                                                             is_s_padded,
                                                                             causal_mask,
                                                                             sm,
                                                                             getStream(),
                                                                             cublas_wrapper_,
                                                                             allocator_,
                                                                             false,
                                                                             has_alibi,
                                                                             scal_alibi,
                                                                             tp_size,
                                                                             tp_rank);
    // cout << "FusedFlashAttentionLayerV2 created" << endl;
}

template<typename T>
FluxSingleAttentionProcessor<T>::FluxSingleAttentionProcessor(FluxSingleAttentionProcessor<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    embedding_dim_(other.embedding_dim_),
    embedding_head_num_(other.embedding_head_num_),
    embedding_head_dim_(other.embedding_head_dim_)
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

    flash_attn_layer = other.flash_attn_layer;
}

template<typename T>
void FluxSingleAttentionProcessor<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxSingleAttentionProcessor::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxSingleAttentionProcessor<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t qkv_buffer_size = sizeof(T) * batch_size * seq_len * 3 * embedding_dim_;

    qkv_buffer1 = (T*)allocator_->reMallocWithName("FluxSingleAttentionProcessor_qkv_buffer1", qkv_buffer_size, false);
    qkv_buffer2 = (T*)allocator_->reMallocWithName("FluxSingleAttentionProcessor_qkv_buffer2", qkv_buffer_size, false);
}

template<typename T>
void FluxSingleAttentionProcessor<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

// 所有Flux层的AdaLayerNorm都复用这个Layer，通过 embedding_scale 这个数值判断不同ada layernorm 需要返回的数据量
template<typename T>
void FluxSingleAttentionProcessor<T>::forward(const TensorMap*                             output_tensors,
                                              const TensorMap*                             input_tensors,
                                              const FluxSingleAttentionProcessorWeight<T>* weights)
{
    // 这里默认temb 已经被silu过了，因为flux里面所有ada layernorm 层都有对temb的前置silu，这里放到外面做了
    // input_tensor       -> [B, S, N * D]
    // temb_tensor        -> [B, N * D]
    // output_tensor      -> [B, S, N * D]
    // msa_output_tensor  -> [embedding_scale, B, N * D]
    Tensor input_tensor    = input_tensors->at("input");
    Tensor rope_emb_tensor = input_tensors->at("rope_emb");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t seq_len    = input_tensor.shape[1];
    // cout << "before FluxSingleAttentionProcessor allocateBuffer" << endl;

    allocateBuffer(batch_size, seq_len);
    // cout << "FluxSingleAttentionProcessor allocateBuffer" << endl;

    int m_1 = batch_size * seq_len;
    int n_1 = embedding_dim_ * 3;
    int k_1 = embedding_dim_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                       // m
                                             m_1,                       // n
                                             k_1,                       // k
                                             weights->to_qkv_weight,    // A
                                             k_1,                       // LDA
                                             input_tensor.getPtr<T>(),  // B
                                             k_1,                       // LDB
                                             qkv_buffer1,               // C
                                             n_1,                       // LDC
                                             weights->to_qkv_bias,      // bias
                                             nullptr,                   // residual
                                             1.0f,                      // alpha
                                             0.0f);                     // beta

    Tensor to_qkv = Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, 3, embedding_dim_}, qkv_buffer1);
    // to_qkv.saveNpy("to_qkv.npy");

    invokeFusedRmsNormAndRope(qkv_buffer2,
                              qkv_buffer1,
                              weights->qk_norm_weight,
                              rope_emb_tensor.getPtr<T>(),
                              1e-6,
                              batch_size,
                              seq_len,
                              embedding_head_num_,
                              embedding_head_dim_,
                              stream_);

    Tensor rope_qkv = Tensor(
        MEMORY_GPU, input_tensor.type, {batch_size, seq_len, 3, embedding_head_num_, embedding_head_dim_}, qkv_buffer2);

    // rope_qkv.saveNpy("rope_qkv.npy");

    std::vector<Tensor> self_attn_input  = {Tensor(MEMORY_GPU,
                                                  input_tensor.type,
                                                   {batch_size, seq_len, 3, embedding_head_num_, embedding_head_dim_},
                                                  qkv_buffer2)};
    std::vector<Tensor> self_attn_output = {Tensor(MEMORY_GPU,
                                                   input_tensor.type,
                                                   {batch_size, seq_len, embedding_head_num_, embedding_head_dim_},
                                                   output_tensor.getPtr<T>())};

    // printf("BasicTransformer , use flashatten2, call flash_attn_layer.\n");
    flash_attn_layer->forward(&self_attn_output, &self_attn_input);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxSingleAttentionProcessor<T>::~FluxSingleAttentionProcessor()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete flash_attn_layer;
}

template class FluxSingleAttentionProcessor<float>;
template class FluxSingleAttentionProcessor<half>;
#ifdef ENABLE_BF16
template class FluxSingleAttentionProcessor<__nv_bfloat16>;
#endif
}  // namespace lyradiff