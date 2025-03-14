#pragma once

#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/attention/cross_fused_attention/fmhca.h"
#include "src/lyradiff/layers/attention/flash_fused_attention_v1/fmha.h"
#include "src/lyradiff/layers/attention/flash_fused_attention_v2/fmha.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlockWeight.h"
#include "src/lyradiff/layers/flash_attn2/flash_attn2.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class BasicTransformerBlock: public BaseLayer {
public:
    // block params
    size_t dim_;
    size_t num_attention_heads_;
    size_t attention_head_dim_;
    size_t cross_attention_dim_;

    size_t ffn_inner_dim1_;
    size_t ffn_inner_dim2_;

    size_t cur_batch           = 0;
    size_t cur_seq_len         = 0;
    size_t cur_encoder_seq_len = 0;

    void   allocateBuffer() override;
    void   freeBuffer() override;
    void   allocateBuffer(size_t batch_size, size_t seq_len, size_t encoder_seq_len, size_t ip_encoder_seq_len);
    size_t getTotalEncodeSeqLenForAllocBuff(TensorMap* input_map, const BasicTransformerBlockWeight<T>* weights);

protected:
    lyradiff::cross_attn::FusedCrossAttentionLayer<T>*   cross_attention_layer;
    lyradiff::flash_attn::FusedFlashAttentionLayerV1<T>* self_attention_layer;
    lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>* flash_attn_layer;
    lyradiff::flash_attn2::FlashAttention2Layer<T>*      flash_attn2_layer;

public:
    T* self_attn_qkv_buf_         = nullptr;
    T* self_attn_qkv_buf2_        = nullptr;
    T* cross_attn_q_buf_          = nullptr;
    T* cross_attn_kv_buf_         = nullptr;
    T* shared_cross_attn_kv_buf2_ = nullptr;
    T* cache_cross_attn_kv_buf2_  = nullptr;
    T* cross_attn_ip_kv_buf_      = nullptr;

    T* norm_hidden_state_buf_ = nullptr;
    T* attn_output_buf_       = nullptr;
    T* attn_output_buf2_      = nullptr;

    // for ffn part
    T* ffn_inter_buf1_ = nullptr;
    T* ffn_inter_buf2_ = nullptr;
    T* ffn_output_buf_ = nullptr;

    bool use_flash_attn_2 = true;
    bool use_fused_geglu  = false;
    // bool use_kv_cache       = false;
    bool is_maintain_ip_buf = false;

    BasicTransformerBlock(size_t           dim,
                          size_t           num_attention_heads,
                          size_t           attention_head_dim,
                          size_t           cross_attention_dim,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward);

    BasicTransformerBlock(BasicTransformerBlock<T> const& basic_transformer_block);

    virtual ~BasicTransformerBlock();

    virtual void forward(std::vector<lyradiff::Tensor>*          output_tensors,
                         const std::vector<lyradiff::Tensor>*    input_tensors,
                         const BasicTransformerBlockWeight<T>* weights);
    virtual void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const BasicTransformerBlockWeight<T>* weights);

    std::unordered_map<std::string, T*> map_alloced_buff_;
};

}  // namespace lyradiff
