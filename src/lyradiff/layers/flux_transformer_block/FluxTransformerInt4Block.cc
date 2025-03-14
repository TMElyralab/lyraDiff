#include "FluxTransformerInt4Block.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxTransformerInt4Block<T>::FluxTransformerInt4Block(size_t              embedding_dim,
                                                      size_t              embedding_head_num,
                                                      size_t              embedding_head_dim,
                                                      size_t              mlp_scale,
                                                      cudaStream_t        stream,
                                                      cublasMMWrapper*    cublas_wrapper,
                                                      IAllocator*         allocator,
                                                      const bool          is_free_buffer_after_forward,
                                                      const bool          sparse,
                                                      const LyraQuantType quant_level):
    FluxTransformerBlock<T>(embedding_dim,
                            embedding_head_num,
                            embedding_head_dim,
                            mlp_scale,
                            stream,
                            cublas_wrapper,
                            allocator,
                            is_free_buffer_after_forward,
                            sparse,
                            quant_level)
{
    if (std::is_same<T, half>::value) {
        printf("set cublas_wrapper to fp16 mode\n");
        this->cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        this->cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to bf16 mode\n");
        this->cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
    // cout << "AdaLayerNorm" << endl;
    // cout << "embedding_dim_: " << embedding_dim_ << endl;
    // this->quant_level_ = LyraQuantType::FP8_W8A8;

    if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL) {
        ada_norm = new AdaFP8LayerNorm<T>(this->embedding_dim_,
                                          6,
                                          false,
                                          this->stream_,
                                          cublas_wrapper,
                                          allocator,
                                          is_free_buffer_after_forward,
                                          sparse);
    }
    else {
        ada_norm = new AdaLayerNorm<T>(this->embedding_dim_,
                                       6,
                                       false,
                                       this->stream_,
                                       cublas_wrapper,
                                       allocator,
                                       is_free_buffer_after_forward,
                                       sparse);
    }

    attn_processor_int4 = new FluxAttentionInt4Processor<T>(this->embedding_dim_,
                                                            this->embedding_head_num_,
                                                            this->embedding_head_dim_,
                                                            this->stream_,
                                                            cublas_wrapper,
                                                            allocator,
                                                            is_free_buffer_after_forward,
                                                            sparse);

    post_processor_int4 = new FluxAttnPostInt4Processor<T>(
        this->embedding_dim_, this->stream_, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse);
}

template<typename T>
FluxTransformerInt4Block<T>::FluxTransformerInt4Block(FluxTransformerInt4Block<T> const& other):
    FluxTransformerBlock<T>(other.embedding_dim_,
                            other.embedding_head_num_,
                            other.embedding_head_dim_,
                            other.mlp_scale_,
                            other.stream_,
                            other.cublas_wrapper_,
                            other.allocator_,
                            other.is_free_buffer_after_forward_,
                            other.sparse_,
                            other.quant_level_)
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

    ada_norm            = other.ada_norm;
    attn_processor_int4 = other.attn_processor_int4;
    post_processor_int4 = other.post_processor_int4;
}

template<typename T>
void FluxTransformerInt4Block<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxTransformerInt4Block::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxTransformerInt4Block<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t context_seq_len)
{
    size_t hidden_buffer_size         = sizeof(T) * batch_size * seq_len * this->embedding_dim_;
    size_t context_hidden_buffer_size = sizeof(T) * batch_size * context_seq_len * this->embedding_dim_;

    size_t msa_buffer_size = sizeof(T) * batch_size * 6 * this->embedding_dim_;

    norm_buffer = (T*)this->allocator_->reMallocWithName("FluxTransformerBlock_norm_buffer", hidden_buffer_size, false);
    context_norm_buffer = (T*)this->allocator_->reMallocWithName(
        "FluxTransformerBlock_context_norm_buffer", context_hidden_buffer_size, false);

    msa_buffer = (T*)this->allocator_->reMallocWithName("FluxTransformerBlock_msa_buffer", msa_buffer_size, false);
    context_msa_buffer =
        (T*)this->allocator_->reMallocWithName("FluxTransformerBlock_context_msa_buffer", msa_buffer_size, false);

    attn_output_buffer =
        (T*)this->allocator_->reMallocWithName("FluxTransformerBlock_attn_output_buffer", hidden_buffer_size, false);

    context_attn_output_buffer = (T*)this->allocator_->reMallocWithName(
        "FluxTransformerBlock_context_attn_output_buffer", context_hidden_buffer_size, false);

    // msa_buffer  = norm_buffer2;
}

template<typename T>
void FluxTransformerInt4Block<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void FluxTransformerInt4Block<T>::forward(const TensorMap*                         output_tensors,
                                          const TensorMap*                         input_tensors,
                                          const FluxTransformerInt4BlockWeight<T>* weights)
{
    Tensor input_tensor         = input_tensors->at("input");
    Tensor context_input_tensor = input_tensors->at("encoder_input");
    Tensor rope_emb_tensor      = input_tensors->at("rope_emb");
    Tensor temb_tensor          = input_tensors->at("temb");

    Tensor output_tensor         = output_tensors->at("output");
    Tensor context_output_tensor = output_tensors->at("encoder_output");

    size_t batch_size      = input_tensor.shape[0];
    size_t seq_len         = input_tensor.shape[1];
    size_t context_seq_len = context_input_tensor.shape[1];

    allocateBuffer(batch_size, seq_len, context_seq_len);

    Tensor norm_hidden_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, norm_buffer);

    Tensor msa_tensor = Tensor(MEMORY_GPU, input_tensor.type, {6, batch_size, this->embedding_dim_}, msa_buffer);

    Tensor attn_output_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, attn_output_buffer);

    Tensor context_norm_hidden_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, context_seq_len, this->embedding_dim_}, context_norm_buffer);

    Tensor context_msa_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {6, batch_size, this->embedding_dim_}, context_msa_buffer);

    Tensor context_attn_output_tensor = Tensor(
        MEMORY_GPU, input_tensor.type, {batch_size, context_seq_len, this->embedding_dim_}, context_attn_output_buffer);

    // temb_tensor.saveNpy("temb_tensor.npy");

    // 开始推理
    TensorMap input_map  = TensorMap({{"input", input_tensor}, {"temb", temb_tensor}});
    TensorMap output_map = TensorMap({{"output", norm_hidden_tensor}, {"msa_output", msa_tensor}});

    // ada_norm->forward(&output_map, &input_map, weights->ada_norm_weight);

    if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL) {
        AdaFP8LayerNorm<T>* block = (AdaFP8LayerNorm<T>*)ada_norm;

        block->forward(&output_map, &input_map, (AdaFP8LayerNormWeight<T>*)weights->ada_norm_weight);
    }
    else {
        ada_norm->forward(&output_map, &input_map, weights->ada_norm_weight);
    }
    // cout << "after ada_norm" << endl;
    // norm_hidden_tensor.saveNpy("norm_hidden_tensor.npy");

    input_map  = TensorMap({{"input", context_input_tensor}, {"temb", temb_tensor}});
    output_map = TensorMap({{"output", context_norm_hidden_tensor}, {"msa_output", context_msa_tensor}});

    // ada_norm->forward(&output_map, &input_map, weights->context_ada_norm_weight);
    if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL) {
        AdaFP8LayerNorm<T>* block = (AdaFP8LayerNorm<T>*)ada_norm;

        block->forward(&output_map, &input_map, (AdaFP8LayerNormWeight<T>*)weights->context_ada_norm_weight);
    }
    else {
        ada_norm->forward(&output_map, &input_map, weights->context_ada_norm_weight);
    }
    // context_norm_hidden_tensor.saveNpy("context_norm_hidden_tensor.npy");

    input_map = TensorMap(
        {{"input", norm_hidden_tensor}, {"encoder_input", context_norm_hidden_tensor}, {"rope_emb", rope_emb_tensor}});
    output_map = TensorMap({{"output", attn_output_tensor}, {"encoder_output", context_attn_output_tensor}});

    attn_processor_int4->forward(&output_map, &input_map, weights->attn_weight);
    // cout << "after attn_processor->forward" << endl;

    // attn_output_tensor.saveNpy("attn_output_tensor.npy");
    // context_attn_output_tensor.saveNpy("context_attn_output_tensor.npy");

    input_map  = TensorMap({{"input", input_tensor}, {"attn_output", attn_output_tensor}, {"msa_input", msa_tensor}});
    output_map = TensorMap({{"output", output_tensor}});

    post_processor_int4->forward(&output_map, &input_map, weights->post_attn_weight);
    // cout << "after post_processor" << endl;

    // output_tensor.saveNpy("output_tensor.npy");

    input_map  = TensorMap({{"input", context_input_tensor},
                            {"attn_output", context_attn_output_tensor},
                            {"msa_input", context_msa_tensor}});
    output_map = TensorMap({{"output", context_output_tensor}});

    post_processor_int4->forward(&output_map, &input_map, weights->context_post_attn_weight);
    // cout << "after post_processor2 " << endl;

    if (this->is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxTransformerInt4Block<T>::~FluxTransformerInt4Block()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
    delete attn_processor_int4;
    delete post_processor_int4;
    delete ada_norm;
}

template class FluxTransformerInt4Block<float>;
template class FluxTransformerInt4Block<half>;
#ifdef ENABLE_BF16
template class FluxTransformerInt4Block<__nv_bfloat16>;
#endif
}  // namespace lyradiff