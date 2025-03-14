#include "FluxSingleTransformerBlock.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/flux_single_transformer_block/flux_single_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/rope/flux_apply_rope.h"
#include "src/lyradiff/utils/cuda_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
FluxSingleTransformerBlock<T>::FluxSingleTransformerBlock(size_t           embedding_dim,
                                                          size_t           embedding_head_num,
                                                          size_t           embedding_head_dim,
                                                          size_t           mlp_scale,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          const bool       is_free_buffer_after_forward,
                                                          const bool       sparse,
                                                          LyraQuantType    quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    embedding_dim_(embedding_dim),
    embedding_head_num_(embedding_head_num),
    embedding_head_dim_(embedding_head_dim),
    mlp_scale_(mlp_scale)
{
    if (std::is_same<T, half>::value) {
        printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to bf16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
    // cout << "AdaLayerNorm" << endl;
    // cout << "embedding_dim_: " << embedding_dim_ << endl;
    ada_norm = new AdaLayerNorm<T>(
        embedding_dim_, 3, false, stream_, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse);
    attn_processor = new FluxSingleAttentionProcessor<T>(embedding_dim_,
                                                         embedding_head_num_,
                                                         embedding_head_dim_,
                                                         stream_,
                                                         cublas_wrapper,
                                                         allocator,
                                                         is_free_buffer_after_forward,
                                                         sparse);
}

template<typename T>
FluxSingleTransformerBlock<T>::FluxSingleTransformerBlock(FluxSingleTransformerBlock<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    embedding_dim_(other.embedding_dim_),
    embedding_head_num_(other.embedding_head_num_),
    embedding_head_dim_(other.embedding_head_dim_),
    mlp_scale_(other.mlp_scale_)
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

    ada_norm       = other.ada_norm;
    attn_processor = other.attn_processor;
}

template<typename T>
void FluxSingleTransformerBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxSingleTransformerBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxSingleTransformerBlock<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t hidden_buffer_size = sizeof(T) * batch_size * seq_len * embedding_dim_;
    size_t msa_buffer_size    = sizeof(T) * batch_size * 3 * embedding_dim_;
    size_t mlp_buffer_size1   = sizeof(T) * batch_size * seq_len * mlp_scale_ * embedding_dim_;
    size_t mlp_buffer_size2   = sizeof(T) * batch_size * seq_len * (mlp_scale_ + 1) * embedding_dim_;

    norm_buffer = (T*)allocator_->reMallocWithName("FluxSingleTransformerBlock_norm_buffer", hidden_buffer_size, false);
    msa_buffer  = (T*)allocator_->reMallocWithName("FluxSingleTransformerBlock_msa_buffer", msa_buffer_size, false);
    attn_output_buffer =
        (T*)allocator_->reMallocWithName("FluxSingleTransformerBlock_attn_output_buffer", hidden_buffer_size, false);
    mlp_buffer1 = (T*)allocator_->reMallocWithName("FluxSingleTransformerBlock_mlp_buffer1", mlp_buffer_size1, false);
    mlp_buffer2 = (T*)allocator_->reMallocWithName("FluxSingleTransformerBlock_mlp_buffer2", mlp_buffer_size2, false);
    // msa_buffer  = norm_buffer2;
}

template<typename T>
void FluxSingleTransformerBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void FluxSingleTransformerBlock<T>::forward(const TensorMap*                           output_tensors,
                                            const TensorMap*                           input_tensors,
                                            const FluxSingleTransformerBlockWeight<T>* weights)
{
    Tensor input_tensor    = input_tensors->at("input");
    Tensor rope_emb_tensor = input_tensors->at("rope_emb");
    Tensor temb_tensor     = input_tensors->at("temb");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t seq_len    = input_tensor.shape[1];

    allocateBuffer(batch_size, seq_len);

    Tensor norm_hidden_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, norm_buffer);

    Tensor msa_tensor = Tensor(MEMORY_GPU, input_tensor.type, {3, batch_size, embedding_dim_}, msa_buffer);

    Tensor attn_output_tensor =
        Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, attn_output_buffer);

    // temb_tensor.saveNpy("temb_tensor.npy");

    // 开始推理
    TensorMap input_map  = TensorMap({{"input", input_tensor}, {"temb", temb_tensor}});
    TensorMap output_map = TensorMap({{"output", norm_hidden_tensor}, {"msa_output", msa_tensor}});

    ada_norm->forward(&output_map, &input_map, weights->ada_norm_weight);
    // cout << "after ada_norm->forward" << endl;

    // msa_tensor.saveNpy("emb_tensor.npy");
    // norm_hidden_tensor.saveNpy("norm_hidden_tensor.npy");

    input_map  = TensorMap({{"input", norm_hidden_tensor}, {"rope_emb", rope_emb_tensor}});
    output_map = TensorMap({{"output", attn_output_tensor}});

    attn_processor->forward(&output_map, &input_map, weights->attn_weight);
    // cout << "after attn_processor->forward" << endl;

    // attn_output_tensor.saveNpy("single_attn_output_tensor.npy");

    T* gate_buffer = &msa_buffer[2 * batch_size * embedding_dim_];

    int m_1 = batch_size * seq_len;
    int n_1 = embedding_dim_ * mlp_scale_;
    int k_1 = embedding_dim_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                       // m
                                             m_1,                       // n
                                             k_1,                       // k
                                             weights->proj_mlp_weight,  // A
                                             k_1,                       // LDA
                                             norm_buffer,               // B
                                             k_1,                       // LDB
                                             mlp_buffer1,               // C
                                             n_1,                       // LDC
                                             weights->proj_mlp_bias,    // bias
                                             nullptr,                   // residual
                                             1.0f,                      // alpha
                                             0.0f);                     // beta

    // Tensor proj_mlp_tensor =
    //     Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_ * mlp_scale_}, mlp_buffer1);

    // proj_mlp_tensor.saveNpy("proj_mlp_tensor.npy");
    // fused_cat_attn_out_and_act_mlp -> mlp_buffer2

    invokeFusedCatAndGelu(mlp_buffer2,
                          attn_output_buffer,
                          mlp_buffer1,
                          batch_size,
                          seq_len,
                          embedding_dim_,
                          embedding_dim_ * mlp_scale_,
                          stream_);

    // Tensor proj_mlp_cat_tensor =
    //     Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_ * (mlp_scale_ + 1)}, mlp_buffer2);
    // proj_mlp_cat_tensor.saveNpy("proj_mlp_cat_tensor.npy");

    m_1 = batch_size * seq_len;
    n_1 = embedding_dim_;
    k_1 = embedding_dim_ * (mlp_scale_ + 1);

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                       // m
                                             m_1,                       // n
                                             k_1,                       // k
                                             weights->proj_out_weight,  // A
                                             k_1,                       // LDA
                                             mlp_buffer2,               // B
                                             k_1,                       // LDB
                                             norm_buffer,               // C
                                             n_1,                       // LDC
                                             weights->proj_out_bias,    // bias
                                             nullptr,                   // residual
                                             1.0f,                      // alpha
                                             0.0f);                     // beta

    Tensor proj_out_tensor = Tensor(MEMORY_GPU, input_tensor.type, {batch_size, seq_len, embedding_dim_}, norm_buffer);
    // proj_out_tensor.saveNpy("proj_out_tensor.npy");

    invokeFusedGateAndResidual(output_tensor.getPtr<T>(),
                               norm_buffer,
                               gate_buffer,
                               input_tensor.getPtr<T>(),
                               batch_size,
                               seq_len,
                               embedding_dim_,
                               stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxSingleTransformerBlock<T>::~FluxSingleTransformerBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete attn_processor;
    delete ada_norm;
}

template class FluxSingleTransformerBlock<float>;
template class FluxSingleTransformerBlock<half>;
#ifdef ENABLE_BF16
template class FluxSingleTransformerBlock<__nv_bfloat16>;
#endif
}  // namespace lyradiff