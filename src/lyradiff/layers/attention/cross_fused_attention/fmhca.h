#pragma once

#include "fmha_cross_attention.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
namespace cross_attn {
int32_t runFMHCAKernel(void const*                               devQ,
                       void const*                               devKV,
                       void*                                     cuSeqlensQ,
                       void*                                     cuSeqlensKV,
                       void*                                     devOutput,
                       int32_t                                   sm,
                       FusedMultiHeadCrossAttentionKernel const* kernels,
                       int32_t                                   b      = 2,
                       int32_t                                   h      = 8,
                       int32_t                                   d      = 64,
                       int32_t                                   seqQ   = 4096,
                       int32_t                                   seqKV  = 77,
                       cudaStream_t                              stream = 0);

template<typename T>
class FusedCrossAttentionLayer: public BaseLayer {
private:
    const size_t hidden_units_;
    const size_t head_num_;

    const int32_t sm_;

    const FusedMultiHeadCrossAttentionKernel* kernels_;

    int32_t* q_cu_seq_lens_;
    int32_t* kv_cu_seq_lens_;

    size_t pre_q_seq_len_  = 0;
    size_t pre_kv_seq_len_ = 0;
    size_t pre_batch_size_ = 0;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t q_seq_len, size_t kv_seq_len);

public:
    FusedCrossAttentionLayer(size_t           head_num,
                             size_t           hidden_units,
                             int32_t          sm,
                             cudaStream_t     stream,
                             cublasMMWrapper* cublas_wrapper,
                             IAllocator*      allocator,
                             bool             is_free_buffer_after_forward);

    virtual ~FusedCrossAttentionLayer();

    virtual void forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors);

    // input_tensor: 必须有 q_buf 键 和 kv_buf 代表输入
    //      q_buf 应该是 [B, QSeqlen, NumHead, PerSizeHead] 拍平的显存数据
    //      kv_buf 应该是 [B, KVSeqlen, NumHead, 2, PersizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLen, NumHead, PerSizeHead] 拍平的显存数据
    virtual void forward(TensorMap* output_tensor, TensorMap* input_tensor);
};

}  // namespace cross_attn
}  // namespace lyradiff