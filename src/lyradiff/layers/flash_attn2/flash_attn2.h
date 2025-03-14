#pragma once

#include "flash_attn2_fwd.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

namespace flash_attn2 {

template<typename T>
class FlashAttention2Layer: public BaseLayer {
private:
    const size_t hidden_units_;
    const size_t head_num_;

    bool   is_causal_;
    float  softmax_scale_;
    float* softmax_lse_;

    const int32_t sm_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

public:
    FlashAttention2Layer(const size_t     head_num,
                         const size_t     hidden_units,
                         const int32_t    sm,
                         cudaStream_t     stream,
                         cublasMMWrapper* cublas_wrapper,
                         IAllocator*      allocator,
                         bool             is_free_buffer_after_forward,
                         const bool       is_causal = false);

    virtual ~FlashAttention2Layer();

    virtual void forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors);

    // input_tensor: 必须有 qkv_buf 键代表输入
    //      q_buf 应该是 [B, SeqlenQ, NumHeadQ, PerSizeHead] 拍平的显存数据
    //      k_buf 应该是 [B, SeqlenK, NumHeadK, PerSizeHead] 拍平的显存数据
    //      v_buf 应该是 [B, SeqlenV, NumHeadV, PerSizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLenQ, NumHeadQ, PerSizeHead] 拍平的显存数据
    virtual void forward(TensorMap* output_tensor, TensorMap* input_tensor);
};
}  // namespace flash_attn2
}  // namespace lyradiff
