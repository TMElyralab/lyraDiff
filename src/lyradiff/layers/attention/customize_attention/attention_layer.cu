
#include "attention.h"
#include "attention_layer.h"

namespace lyradiff {

template<OperationType OpType>
AttentionLayer<OpType>::AttentionLayer(const int              max_batch_size,
                                       const int              head_num,
                                       const int              size_per_head,
                                       const int              max_seq_len,
                                       AttentionParam<OpType> param,
                                       bool                   use_fused_attention):
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    use_fused_attention_(use_fused_attention)
{
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    arch_ = major * 10 + minor;
    if (arch_ < 70)
        use_fused_attention_ = false;

    // different performance of fused attention due to bandwidth of different
    // devices
    if (use_fused_attention_ && (max_seq_len_ % 2 == 0)) {
        if (arch_ == 70 && max_seq_len_ > 256)  // V100
            use_fused_attention_ = false;

        else if (arch_ == 80 && max_seq_len_ > 384)  // A100
            use_fused_attention_ = false;

        else if (arch_ == 86 && max_seq_len_ > 256)
            use_fused_attention_ = false;

        else if (arch_ > 86 && max_seq_len_ > 128)
            use_fused_attention_ = false;
    }

#ifdef CUTLASS_ATTENTION
    bool use_cutlass       = false;
    using CutlassAttention = cutlass_ops::CutlassAttention<OpType>;
    if (arch_ == 80 && OpType == OperationType::HALF && CutlassAttention::check_seqlen_supported(max_seq_len)) {
        use_cutlass = true;
    }

    if (use_cutlass) {
        if constexpr (OpType == OperationType::HALF) {
            attention_layer_ =
                new CutlassAttention(max_batch_size_, head_num_, size_per_head_, max_seq_len_, use_fused_attention_);
        }
        else {
            throw std::logic_error("Only half supported");
        }
    }
    else {
#else
    {
#endif  // CUTLASS_ATTENTION
        attention_layer_ =
            new Attention<OpType>(max_batch_size, head_num, size_per_head, max_seq_len, use_fused_attention_);
    }

    attention_layer_->initialize(param);
}

template<OperationType OpType>
void AttentionLayer<OpType>::infer(AttentionInferParam infer_param)
{
    // 目前实现的推理根据序列的长度，分为了几类，以适应不同长度区间下不同的 CUDA 优化策略

    //  1. 长于 352 的用 no fused 的方式计算，但是也对最大长度有限制，因为共享内存的原因：
    //      要求 num_heads*seq_len*sizeof(float) 不能超过 64*1024
    //      在 sd 中， num_heads=8, sizeof(float)=4, 那么 seq_len 最大等于 64*1024/8/4 = 2048
    //  2. 长度大于 80，小于 352：
    //      用 attention_fused_long.cu 中实现的长序列下的 fused attention 算法，利用 tensorcore 计算
    //  3. 长度短于 80:
    //      使用用attention_fused_short.cu 中实现的短序列下的 fused attention 算法，利用 tensorcore 计算

    attention_layer_->infer(infer_param);
}
}  // namespace lyradiff