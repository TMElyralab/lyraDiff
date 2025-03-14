#pragma once

#include "fmhaRunner.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

namespace flash_attn {

template<typename T>
class FusedFlashAttentionLayerV2: public BaseLayer {
private:
    const size_t hidden_units_;
    const size_t head_num_;
    const size_t kv_head_num_;
    const float  q_scaling_;
    const bool   causal_mask_;

    const int32_t sm_;

    const bool has_alibi_   = false;
    const bool scale_alibi_ = false;
    const int  tp_size_     = 1;
    const int  tp_rank_     = 0;

    int32_t* cu_seq_lens_;

    size_t pre_seq_len_    = 0;
    size_t pre_batch_size_ = 0;

    FusedMHARunnerV2* runner_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

public:
    FusedFlashAttentionLayerV2(size_t           head_num,
                               size_t           hidden_units,
                               size_t           kv_head_num,
                               float            q_scaling,
                               const bool       force_fp32_acc,
                               const bool       is_s_padded,
                               const bool       causal_mask,
                               int32_t          sm,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               bool             is_free_buffer_after_forward,
                               const bool       has_alibi   = false,
                               const bool       scale_alibi = false,
                               const int        tp_size     = 1,
                               const int        tp_rank     = 0);

    virtual ~FusedFlashAttentionLayerV2();

    virtual void forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors);

    // input_tensor: 必须有 qkv_buf 键代表输入
    //      qkv_buf 应该是 [B, Seqlen, 3, NumHead, PerSizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLen, NumHead, PerSizeHead] 拍平的显存数据
    virtual void forward(TensorMap* output_tensor, TensorMap* input_tensor);
};
}  // namespace flash_attn
}  // namespace lyradiff
