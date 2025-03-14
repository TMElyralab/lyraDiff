#include "FluxAttentionInt4Processor.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/flux_transformer_block/flux_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxAttentionInt4Processor<T>::FluxAttentionInt4Processor(size_t           embedding_dim,
                                                          size_t           embedding_head_num,
                                                          size_t           embedding_head_dim,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          const bool       is_free_buffer_after_forward,
                                                          const bool       sparse):
    FluxAttentionProcessor<T>(embedding_dim,
                              embedding_head_num,
                              embedding_head_dim,
                              stream,
                              cublas_wrapper,
                              allocator,
                              is_free_buffer_after_forward,
                              sparse)
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

    flash_attn_layer = new lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>(this->embedding_head_num_,
                                                                             this->embedding_head_dim_,
                                                                             this->embedding_head_num_,
                                                                             q_scaling,
                                                                             force_fp32_acc,
                                                                             is_s_padded,
                                                                             causal_mask,
                                                                             sm,
                                                                             this->getStream(),
                                                                             this->cublas_wrapper_,
                                                                             this->allocator_,
                                                                             false,
                                                                             has_alibi,
                                                                             scal_alibi,
                                                                             tp_size,
                                                                             tp_rank);

    to_qkv_gemm = new W4A4Gemm<T>(embedding_dim * 3,
                                  embedding_dim,
                                  32,
                                  true,
                                  64,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward,
                                  sparse);

    to_out_gemm = new W4A4Gemm<T>(embedding_dim,
                                  embedding_dim,
                                  32,
                                  true,
                                  64,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward,
                                  sparse);
    // cout << "FusedFlashAttentionLayerV2 created" << endl;
}

template<typename T>
FluxAttentionInt4Processor<T>::FluxAttentionInt4Processor(FluxAttentionInt4Processor<T> const& other):
    FluxAttentionProcessor<T>(other.embedding_dim_,
                              other.embedding_head_num_,
                              other.embedding_head_dim_,
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

    flash_attn_layer = other.flash_attn_layer;
}

template<typename T>
void FluxAttentionInt4Processor<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxAttentionInt4Processor::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxAttentionInt4Processor<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t encoder_seq_len)
{
    size_t qkv_buffer_size         = sizeof(T) * batch_size * seq_len * 3 * this->embedding_dim_;
    size_t encoder_qkv_buffer_size = sizeof(T) * batch_size * encoder_seq_len * 3 * this->embedding_dim_;
    size_t nhidden_buffer_size     = sizeof(T) * batch_size * (seq_len + encoder_seq_len) * 3 * this->embedding_dim_;
    size_t attn_output_buffer_size = sizeof(T) * batch_size * (seq_len + encoder_seq_len) * this->embedding_dim_;
    // size_t fp8_buffer_size = sizeof(__nv_fp8_e4m3) * batch_size * (seq_len + encoder_seq_len) * this->embedding_dim_;

    qkv_buffer = (T*)this->allocator_->reMallocWithName("FluxAttentionProcessor_qkv_buffer", qkv_buffer_size, false);
    encoder_qkv_buffer = (T*)this->allocator_->reMallocWithName(
        "FluxAttentionProcessor_encoder_qkv_buffer", encoder_qkv_buffer_size, false);
    nhidden_buffer =
        (T*)this->allocator_->reMallocWithName("FluxAttentionProcessor_nhidden_buffer", nhidden_buffer_size, false);
    attn_out_buffer = (T*)this->allocator_->reMallocWithName(
        "FluxAttentionProcessor_attn_out_buffer", attn_output_buffer_size, false);
    // fp8_buffer = (__nv_fp8_e4m3*)this->allocator_->reMallocWithName("fp8_input_buffer", fp8_buffer_size, false);
}

template<typename T>
void FluxAttentionInt4Processor<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

// 所有Flux层的AdaLayerNorm都复用这个Layer，通过 embedding_scale 这个数值判断不同ada layernorm 需要返回的数据量
template<typename T>
void FluxAttentionInt4Processor<T>::forward(const TensorMap*                           output_tensors,
                                            const TensorMap*                           input_tensors,
                                            const FluxAttentionInt4ProcessorWeight<T>* weights)
{
    // 这里默认temb 已经被silu过了，因为flux里面所有ada layernorm 层都有对temb的前置silu，这里放到外面做了
    // input_tensor       -> [B, S, N * D]
    // temb_tensor        -> [B, N * D]
    // output_tensor      -> [B, S, N * D]
    // msa_output_tensor  -> [embedding_scale, B, N * D]
    Tensor input_tensor         = input_tensors->at("input");
    Tensor encoder_input_tensor = input_tensors->at("encoder_input");
    Tensor rope_emb_tensor      = input_tensors->at("rope_emb");

    Tensor output_tensor         = output_tensors->at("output");
    Tensor encoder_output_tensor = output_tensors->at("encoder_output");

    size_t batch_size      = input_tensor.shape[0];
    size_t seq_len         = input_tensor.shape[1];
    size_t encoder_seq_len = encoder_input_tensor.shape[1];
    // cout << "before FluxSingleAttentionProcessor allocateBuffer" << endl;

    allocateBuffer(batch_size, seq_len, encoder_seq_len);
    // cout << "FluxSingleAttentionProcessor allocateBuffer" << endl;

    Tensor qkv_input =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_}, input_tensor.getPtr<T>());

    Tensor qkv = Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, 3 * this->embedding_dim_}, qkv_buffer);

    Tensor encoder_qkv_input = Tensor(MEMORY_GPU,
                                      input_tensor.type,
                                      {batch_size * encoder_seq_len, this->embedding_dim_},
                                      encoder_input_tensor.getPtr<T>());

    Tensor encoder_qkv = Tensor(
        MEMORY_GPU, input_tensor.type, {batch_size * encoder_seq_len, 3 * this->embedding_dim_}, encoder_qkv_buffer);

    TensorMap input_map  = TensorMap({{"input", qkv_input}});
    TensorMap output_map = TensorMap({{"output", qkv}});

    // input.saveNpy("input.npy");

    to_qkv_gemm->forward(&output_map, &input_map, weights->to_qkv_weight);

    input_map  = TensorMap({{"input", encoder_qkv_input}});
    output_map = TensorMap({{"output", encoder_qkv}});

    to_qkv_gemm->forward(&output_map, &input_map, weights->encoder_to_qkv_weight);

    // qkv.saveNpy("qkv_no_bias.npy");
    // encoder_qkv.saveNpy("encoder_qkv_no_bias.npy");

    invokeAddBias(qkv_buffer,
                  qkv_buffer,
                  weights->to_qkv_weight->bias,
                  batch_size,
                  seq_len,
                  3 * this->embedding_dim_,
                  this->stream_);

    invokeAddBias(encoder_qkv_buffer,
                  encoder_qkv_buffer,
                  weights->encoder_to_qkv_weight->bias,
                  batch_size,
                  encoder_seq_len,
                  3 * this->embedding_dim_,
                  this->stream_);

    // qkv.saveNpy("qkv.npy");
    // encoder_qkv.saveNpy("encoder_qkv.npy");

    invokeFusedRmsNormCatAndRope(nhidden_buffer,
                                 qkv_buffer,
                                 encoder_qkv_buffer,
                                 weights->qk_norm_weight,
                                 weights->encoder_qk_norm_weight,
                                 rope_emb_tensor.getPtr<T>(),
                                 1e-6,
                                 batch_size,
                                 seq_len,
                                 encoder_seq_len,
                                 this->embedding_head_num_,
                                 this->embedding_head_dim_,
                                 this->stream_);

    Tensor cat_qkv =
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, seq_len + encoder_seq_len, 3, this->embedding_head_num_, this->embedding_head_dim_},
               nhidden_buffer);

    // cat_qkv.saveNpy("cat_qkv.npy");
    // Tensor attn_output = Tensor(MEMORY_GPU,
    //                             input_tensor.type,
    //                             {batch_size, (seq_len + encoder_seq_len), embedding_head_num_, embedding_head_dim_},
    //                             nhidden_buffer);

    std::vector<Tensor> self_attn_input = {
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, (seq_len + encoder_seq_len), 3, this->embedding_head_num_, this->embedding_head_dim_},
               nhidden_buffer)};
    std::vector<Tensor> self_attn_output = {
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, (seq_len + encoder_seq_len), this->embedding_head_num_, this->embedding_head_dim_},
               attn_out_buffer)};

    // printf("BasicTransformer , use flashatten2, call flash_attn_layer.\n");
    flash_attn_layer->forward(&self_attn_output, &self_attn_input);

    Tensor attnout =
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, seq_len + encoder_seq_len, this->embedding_head_num_ * this->embedding_head_dim_},
               attn_out_buffer);

    // attnout.saveNpy("attnout.npy");

    // 复用之前的 nhidden_buffer
    T* hidden_ptr         = &nhidden_buffer[input_tensor.size()];
    T* encoder_hidden_ptr = nhidden_buffer;

    // TODO: split input and encoder kernel -> hidden_ptr and encoder_hidden_ptr

    invokeSpiltEncoderAndHidden(hidden_ptr,
                                encoder_hidden_ptr,
                                attn_out_buffer,
                                batch_size,
                                seq_len,
                                encoder_seq_len,
                                this->embedding_dim_,
                                this->stream_);

    Tensor hidden_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_}, hidden_ptr);
    Tensor encoder_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * encoder_seq_len, this->embedding_dim_}, encoder_hidden_ptr);

    // hidden_tensor.saveNpy("hidden_tensor.npy");
    // encoder_tensor.saveNpy("encoder_tensor.npy");

    Tensor hidden_to_out_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_}, output_tensor.getPtr<T>());
    Tensor encoder_to_out_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * encoder_seq_len, this->embedding_dim_}, encoder_output_tensor.getPtr<T>());

    input_map  = TensorMap({{"input", hidden_tensor}});
    output_map = TensorMap({{"output", hidden_to_out_tensor}});

    to_out_gemm->forward(&output_map, &input_map, weights->to_out_weight);

    input_map  = TensorMap({{"input", encoder_tensor}});
    output_map = TensorMap({{"output", encoder_to_out_tensor}});

    to_out_gemm->forward(&output_map, &input_map, weights->encoder_to_out_weight);

    invokeAddBias(output_tensor.getPtr<T>(),
                  output_tensor.getPtr<T>(),
                  weights->to_out_weight->bias,
                  batch_size,
                  seq_len,
                  this->embedding_dim_,
                  this->stream_);

    invokeAddBias(encoder_output_tensor.getPtr<T>(),
                  encoder_output_tensor.getPtr<T>(),
                  weights->encoder_to_out_weight->bias,
                  batch_size,
                  encoder_seq_len,
                  this->embedding_dim_,
                  this->stream_);

    if (this->is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxAttentionInt4Processor<T>::~FluxAttentionInt4Processor()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
    delete flash_attn_layer;
}

template class FluxAttentionInt4Processor<float>;
template class FluxAttentionInt4Processor<half>;
#ifdef ENABLE_BF16
template class FluxAttentionInt4Processor<__nv_bfloat16>;
#endif
}  // namespace lyradiff