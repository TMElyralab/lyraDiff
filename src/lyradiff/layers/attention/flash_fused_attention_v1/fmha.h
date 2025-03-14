#pragma once

#include "fmha_flash_attention.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

namespace flash_attn {
int32_t runFMHFAKernel(void const*                               devQKV,
                       void*                                     cuSeqlens,
                       void*                                     devOutput,
                       size_t                                    total,
                       int32_t                                   sm,
                       FusedMultiHeadFlashAttentionKernel const* kernels,
                       int32_t                                   b      = 2,
                       int32_t                                   h      = 8,
                       int32_t                                   d      = 64,
                       int32_t                                   s      = 4096,
                       cudaStream_t                              stream = 0);

template<typename T>
class FusedFlashAttentionLayerV1: public BaseLayer {
private:
    const size_t hidden_units_;
    const size_t head_num_;

    const int32_t sm_;

    const FusedMultiHeadFlashAttentionKernel* kernels_;

    int32_t* cu_seq_lens_;

    size_t pre_seq_len_    = 0;
    size_t pre_batch_size_ = 0;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

public:
    FusedFlashAttentionLayerV1(size_t           head_num,
                               size_t           hidden_units,
                               int32_t          sm,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               bool             is_free_buffer_after_forward);

    virtual ~FusedFlashAttentionLayerV1();

    virtual void forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors);

    // input_tensor: 必须有 qkv_buf 键代表输入
    //      qkv_buf 应该是 [B, Seqlen, NumHead, 3, PerSizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLen, NumHead, PerSizeHead] 拍平的显存数据
    virtual void forward(TensorMap* output_tensor, TensorMap* input_tensor);
};
}  // namespace flash_attn
}  // namespace lyradiff
