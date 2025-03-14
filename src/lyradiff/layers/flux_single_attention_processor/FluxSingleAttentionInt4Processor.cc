#include "FluxSingleAttentionInt4Processor.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxSingleAttentionInt4Processor<T>::FluxSingleAttentionInt4Processor(size_t           embedding_dim,
                                                                      size_t           embedding_head_num,
                                                                      size_t           embedding_head_dim,
                                                                      cudaStream_t     stream,
                                                                      cublasMMWrapper* cublas_wrapper,
                                                                      IAllocator*      allocator,
                                                                      const bool       is_free_buffer_after_forward,
                                                                      const bool       sparse):
    FluxSingleAttentionProcessor<T>(embedding_dim,
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
    // cout << "FusedFlashAttentionLayerV2 created" << endl;
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
}

template<typename T>
FluxSingleAttentionInt4Processor<T>::FluxSingleAttentionInt4Processor(FluxSingleAttentionInt4Processor<T> const& other):
    FluxSingleAttentionInt4Processor<T>(other.embedding_dim_,
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
void FluxSingleAttentionInt4Processor<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxSingleAttentionFP8Processor::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxSingleAttentionInt4Processor<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t qkv_buffer_size = sizeof(T) * batch_size * seq_len * 3 * this->embedding_dim_;
    // size_t input_buffer_size = sizeof(__nv_fp8_e4m3) * batch_size * seq_len * this->embedding_dim_;

    qkv_buffer1 =
        (T*)this->allocator_->reMallocWithName("FluxSingleAttentionInt4Processor_qkv_buffer1", qkv_buffer_size, false);
    qkv_buffer2 =
        (T*)this->allocator_->reMallocWithName("FluxSingleAttentionInt4Processor_qkv_buffer2", qkv_buffer_size, false);
    // input_buffer = (__nv_fp8_e4m3*)this->allocator_->reMallocWithName("fp8_input_buffer", input_buffer_size, false);
}

template<typename T>
void FluxSingleAttentionInt4Processor<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

// 所有Flux层的AdaLayerNorm都复用这个Layer，通过 embedding_scale 这个数值判断不同ada layernorm 需要返回的数据量
template<typename T>
void FluxSingleAttentionInt4Processor<T>::forward(const TensorMap*                                 output_tensors,
                                                  const TensorMap*                                 input_tensors,
                                                  const FluxSingleAttentionInt4ProcessorWeight<T>* weights)
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
    // cout << "after allocateBuffer" << endl;

    Tensor input =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, this->embedding_dim_}, input_tensor.getPtr<T>());
    Tensor to_qkv =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size * seq_len, 3 * this->embedding_dim_}, qkv_buffer1);

    TensorMap input_map  = TensorMap({{"input", input}});
    TensorMap output_map = TensorMap({{"output", to_qkv}});

    // input.saveNpy("input.npy");

    to_qkv_gemm->forward(&output_map, &input_map, weights->to_qkv_weight);

    invokeFusedBiasRmsNormAndRope(qkv_buffer2,
                                  qkv_buffer1,
                                  weights->to_qkv_weight->bias,
                                  weights->qk_norm_weight,
                                  rope_emb_tensor.getPtr<T>(),
                                  1e-6,
                                  batch_size,
                                  seq_len,
                                  this->embedding_head_num_,
                                  this->embedding_head_dim_,
                                  this->stream_);

    Tensor rope_qkv = Tensor(MEMORY_GPU,
                             input_tensor.type,
                             {batch_size, seq_len, 3, this->embedding_head_num_, this->embedding_head_dim_},
                             qkv_buffer2);

    // rope_qkv.saveNpy("rope_qkv.npy");
    // cout << "rope_qkv.saveNpy" << endl;

    std::vector<Tensor> self_attn_input = {
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, seq_len, 3, this->embedding_head_num_, this->embedding_head_dim_},
               qkv_buffer2)};
    std::vector<Tensor> self_attn_output = {
        Tensor(MEMORY_GPU,
               input_tensor.type,
               {batch_size, seq_len, this->embedding_head_num_, this->embedding_head_dim_},
               output_tensor.getPtr<T>())};

    // printf("BasicTransformer , use flashatten2, call flash_attn_layer.\n");
    flash_attn_layer->forward(&self_attn_output, &self_attn_input);

    if (this->is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxSingleAttentionInt4Processor<T>::~FluxSingleAttentionInt4Processor()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
    delete flash_attn_layer;
}

template class FluxSingleAttentionInt4Processor<float>;
template class FluxSingleAttentionInt4Processor<half>;
#ifdef ENABLE_BF16
template class FluxSingleAttentionInt4Processor<__nv_bfloat16>;
#endif
}  // namespace lyradiff