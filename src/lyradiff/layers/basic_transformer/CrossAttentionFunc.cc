#include "src/lyradiff/layers/basic_transformer/CrossAttentionFunc.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"

namespace lyradiff {

template<typename T>
void CrossAttnProcessorBasicFunc(TensorMap*                            output_tensors,
                                 TensorMap*                            input_tensors,
                                 const BasicTransformerBlockWeight<T>* weights,
                                 BasicTransformerBlock<T>*             basic_transformer_block)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto& norm_hidden_state_buf_ = basic_transformer_block.norm_hidden_state_buf_;
    auto& cross_attn_q_buf_      = basic_transformer_block.cross_attn_q_buf_;
    auto& cross_attn_kv_buf_     = basic_transformer_block->cross_attn_kv_buf_;
    auto& cross_attn_kv_buf2_    = basic_transformer_block->cross_attn_kv_buf2_;
    auto& attn_output_buf_       = basic_transformer_block->attn_output_buf_;

    Tensor init_hidden_states    = input_tensors->at(HIDDEN_STATES);
    Tensor encoder_hidden_states = input_tensors->at(ENCODER_HIDDEN_STATES);
    Tensor input                 = input_tensors->at(NM_INPUT);
    Tensor output                = output_tensors->at(NM_OUTPUT);

    auto encoder_hidden_states_ptr = encoder_hidden_states.getPtr<T>();

    size_t batch_size      = init_hidden_states.shape[0];
    size_t seq_len         = init_hidden_states.shape[1];
    size_t inner_dim       = init_hidden_states.shape[2];
    size_t encoder_seq_len = encoder_hidden_states.shape[1];

    size_t dim_                 = basic_transformer_block->dim_;
    size_t cross_attention_dim_ = basic_transformer_block->dim_;
    size_t num_attention_heads_ = basic_transformer_block->num_attention_heads_;
    size_t attention_head_dim_  = basic_transformer_block->attention_head_dim_;

    auto& cublas_wrapper_       = basic_transformer_block->cublas_wrapper_;
    auto& cross_attention_layer = basic_transformer_block->cross_attention_layer;
    auto& stream                = basic_transformer_block->getStream();

    int m, n, k;
    m = seq_len * batch_size;
    n = dim_;
    k = dim_;

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          weights->attention2_q_weight,
                          k,
                          norm_hidden_state_buf_,
                          k,
                          cross_attn_q_buf_,
                          n);

    m = encoder_seq_len * batch_size;
    n = dim_ * 2;
    k = cross_attention_dim_;

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          weights->attention2_kv_weight,
                          k,
                          encoder_hidden_states_ptr,
                          k,
                          cross_attn_kv_buf_,
                          n);

    invokeCrossAttnKernelInputPermute<T>(cross_attn_kv_buf2_,
                                         cross_attn_kv_buf_,
                                         batch_size,
                                         encoder_seq_len,
                                         dim_,
                                         num_attention_heads_,
                                         attention_head_dim_,
                                         basic_transformer_block->getStream());

    std::vector<Tensor> cross_attn_input = {
        Tensor(MEMORY_GPU,
               init_hidden_states.type,
               {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
               cross_attn_q_buf_),
        Tensor(MEMORY_GPU,
               init_hidden_states.type,
               {batch_size, encoder_seq_len, num_attention_heads_, 2, attention_head_dim_},
               cross_attn_kv_buf2_)};
    std::vector<Tensor> cross_attn_output = {Tensor(MEMORY_GPU,
                                                    init_hidden_states.type,
                                                    {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                                                    attn_output_buf_)};

    cross_attention_layer->forward(&cross_attn_output, &cross_attn_input);
}
}  // namespace lyradiff