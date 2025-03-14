#include "FluxSingleTransformerFP8Block.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxSingleTransformerFP8Block<T>::FluxSingleTransformerFP8Block(size_t           embedding_dim,
                                                                size_t           embedding_head_num,
                                                                size_t           embedding_head_dim,
                                                                size_t           mlp_scale,
                                                                cudaStream_t     stream,
                                                                cublasMMWrapper* cublas_wrapper,
                                                                IAllocator*      allocator,
                                                                const bool       is_free_buffer_after_forward,
                                                                const bool       sparse,
                                                                LyraQuantType    quant_level):
    FluxSingleTransformerBlock<T>(embedding_dim,
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

    if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL) {
        ada_norm = new AdaFP8LayerNorm<T>(this->embedding_dim_,
                                          3,
                                          false,
                                          this->stream_,
                                          cublas_wrapper,
                                          allocator,
                                          is_free_buffer_after_forward,
                                          sparse);
    }
    else {
        ada_norm = new AdaLayerNorm<T>(this->embedding_dim_,
                                       3,
                                       false,
                                       this->stream_,
                                       cublas_wrapper,
                                       allocator,
                                       is_free_buffer_after_forward,
                                       sparse);
    }
    attn_processor = new FluxSingleAttentionFP8Processor<T>(this->embedding_dim_,
                                                            this->embedding_head_num_,
                                                            this->embedding_head_dim_,
                                                            this->stream_,
                                                            cublas_wrapper,
                                                            allocator,
                                                            is_free_buffer_after_forward,
                                                            sparse);
}

template<typename T>
FluxSingleTransformerFP8Block<T>::FluxSingleTransformerFP8Block(FluxSingleTransformerFP8Block<T> const& other):
    FluxSingleTransformerBlock<T>(other.embedding_dim_,
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

    ada_norm       = other.ada_norm;
    attn_processor = other.attn_processor;
}

template<typename T>
void FluxSingleTransformerFP8Block<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxSingleTransformerFP8Block::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxSingleTransformerFP8Block<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t hidden_buffer_size = sizeof(T) * batch_size * seq_len * this->embedding_dim_;
    size_t msa_buffer_size    = sizeof(T) * batch_size * 3 * this->embedding_dim_;
    size_t mlp_buffer_size1   = sizeof(T) * batch_size * seq_len * this->mlp_scale_ * this->embedding_dim_;
    size_t mlp_buffer_size2   = sizeof(T) * batch_size * seq_len * (this->mlp_scale_ + 1) * this->embedding_dim_;

    size_t fp8_buffer_size =
        sizeof(__nv_fp8_e4m3) * batch_size * seq_len * (this->mlp_scale_ + 1) * this->embedding_dim_;

    // T* norm_buffer;
    // T* attn_output_buffer;
    // T* msa_buffer;
    // T* mlp_buffer1;
    // T* mlp_buffer2;

    norm_buffer =
        (T*)this->allocator_->reMallocWithName("FluxSingleTransformerBlock_norm_buffer", hidden_buffer_size, false);
    msa_buffer =
        (T*)this->allocator_->reMallocWithName("FluxSingleTransformerBlock_msa_buffer", msa_buffer_size, false);
    attn_output_buffer = (T*)this->allocator_->reMallocWithName(
        "FluxSingleTransformerBlock_attn_output_buffer", hidden_buffer_size, false);
    mlp_buffer1 =
        (T*)this->allocator_->reMallocWithName("FluxSingleTransformerBlock_mlp_buffer1", mlp_buffer_size1, false);
    mlp_buffer2 =
        (T*)this->allocator_->reMallocWithName("FluxSingleTransformerBlock_mlp_buffer2", mlp_buffer_size2, false);

    fp8_buffer1 = (__nv_fp8_e4m3*)this->allocator_->reMallocWithName("fp8_input_buffer", fp8_buffer_size, false);
    // msa_buffer  = norm_buffer2;
}

template<typename T>
void FluxSingleTransformerFP8Block<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void FluxSingleTransformerFP8Block<T>::forward(const TensorMap*                              output_tensors,
                                               const TensorMap*                              input_tensors,
                                               const FluxSingleTransformerFP8BlockWeight<T>* weights)
{
    Tensor input_tensor    = input_tensors->at("input");
    Tensor rope_emb_tensor = input_tensors->at("rope_emb");
    Tensor temb_tensor     = input_tensors->at("temb");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t seq_len    = input_tensor.shape[1];

    allocateBuffer(batch_size, seq_len);

    Tensor norm_hidden_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, norm_buffer);

    Tensor msa_tensor = Tensor(MEMORY_GPU, input_tensor.type, {3, batch_size, this->embedding_dim_}, msa_buffer);

    Tensor attn_output_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, this->embedding_dim_}, attn_output_buffer);

    // temb_tensor.saveNpy("temb_tensor.npy");

    // 开始推理
    TensorMap input_map  = TensorMap({{"input", input_tensor}, {"temb", temb_tensor}});
    TensorMap output_map = TensorMap({{"output", norm_hidden_tensor}, {"msa_output", msa_tensor}});

    if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL) {
        AdaFP8LayerNorm<T>* block = (AdaFP8LayerNorm<T>*)ada_norm;

        block->forward(&output_map, &input_map, (AdaFP8LayerNormWeight<T>*)weights->ada_norm_weight);
    }
    else {
        ada_norm->forward(&output_map, &input_map, weights->ada_norm_weight);
    }
    // cout << "after ada_norm->forward" << endl;

    input_map  = TensorMap({{"input", norm_hidden_tensor}, {"rope_emb", rope_emb_tensor}});
    output_map = TensorMap({{"output", attn_output_tensor}});

    if (typeid(*attn_processor) == typeid(FluxSingleAttentionFP8Processor<T>)) {
        FluxSingleAttentionFP8Processor<T>* fp8_processor = (FluxSingleAttentionFP8Processor<T>*)attn_processor;
        fp8_processor->forward(
            &output_map, &input_map, (FluxSingleAttentionFP8ProcessorWeight<T>*)weights->attn_weight);
    }
    else {
        attn_processor->forward(&output_map, &input_map, weights->attn_weight);
    }

    T* gate_buffer = &msa_buffer[2 * batch_size * this->embedding_dim_];

    invokeCudaD2DScaleCpyConvert(fp8_buffer1,
                                 norm_buffer,
                                 weights->proj_mlp_input_scale,
                                 true,
                                 batch_size * seq_len * this->embedding_dim_,
                                 this->stream_);

    int m_1 = batch_size * seq_len;
    int n_1 = this->embedding_dim_ * this->mlp_scale_;
    int k_1 = this->embedding_dim_;

    this->cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   n_1,                             // m
                                                   m_1,                             // n
                                                   k_1,                             // k
                                                   weights->proj_mlp_weight,        // A
                                                   k_1,                             // LDA
                                                   weights->proj_mlp_weight_scale,  // weight scale
                                                   fp8_buffer1,                     // B
                                                   k_1,                             // LDB
                                                   weights->proj_mlp_input_scale,   // input scale
                                                   mlp_buffer1,                     // C
                                                   n_1,                             // LDC
                                                   weights->proj_mlp_bias,          // bias
                                                   nullptr,                         // residual
                                                   1.0f,                            // alpha
                                                   0.0f);                           // beta

    invokeFusedCatAndGelu(mlp_buffer2,
                          attn_output_buffer,
                          mlp_buffer1,
                          batch_size,
                          seq_len,
                          this->embedding_dim_,
                          this->embedding_dim_ * this->mlp_scale_,
                          this->stream_);

    invokeCudaD2DScaleCpyConvert(fp8_buffer1,
                                 mlp_buffer2,
                                 weights->proj_out_input_scale,
                                 true,
                                 batch_size * seq_len * this->embedding_dim_ * (this->mlp_scale_ + 1),
                                 this->stream_);

    m_1 = batch_size * seq_len;
    n_1 = this->embedding_dim_;
    k_1 = this->embedding_dim_ * (this->mlp_scale_ + 1);

    this->cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   n_1,                             // m
                                                   m_1,                             // n
                                                   k_1,                             // k
                                                   weights->proj_out_weight,        // A
                                                   k_1,                             // LDA
                                                   weights->proj_out_weight_scale,  // weight scale
                                                   fp8_buffer1,                     // B
                                                   k_1,                             // LDB
                                                   weights->proj_out_input_scale,   // input scale
                                                   norm_buffer,                     // C
                                                   n_1,                             // LDC
                                                   weights->proj_out_bias,          // bias
                                                   nullptr,                         // residual
                                                   1.0f,                            // alpha
                                                   0.0f);                           // beta

    invokeFusedGateAndResidual(output_tensor.getPtr<T>(),
                               norm_buffer,
                               gate_buffer,
                               input_tensor.getPtr<T>(),
                               batch_size,
                               seq_len,
                               this->embedding_dim_,
                               this->stream_);

    if (this->is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxSingleTransformerFP8Block<T>::~FluxSingleTransformerFP8Block()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
    delete attn_processor;
    delete ada_norm;
}

template class FluxSingleTransformerFP8Block<float>;
template class FluxSingleTransformerFP8Block<half>;
#ifdef ENABLE_BF16
template class FluxSingleTransformerFP8Block<__nv_bfloat16>;
#endif
}  // namespace lyradiff