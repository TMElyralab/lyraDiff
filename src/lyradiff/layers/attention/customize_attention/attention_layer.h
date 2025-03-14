#pragma once

#include "attention.h"

namespace lyradiff {

template<OperationType OpType>
struct AttentionLayer {
protected:
    typedef Traits<OpType>             Traits_;
    typedef typename Traits_::DataType DataType_;
    using AttentionInferParam = struct AttentionInferParam<DataType_>;

public:
    const int          max_batch_size_, max_seq_len_, head_num_, size_per_head_;
    bool               use_fused_attention_ = true;
    Attention<OpType>* attention_layer_     = nullptr;
    int                arch_                = 80;

    AttentionLayer(const int              max_batch_size,
                   const int              head_num,
                   const int              size_per_head,
                   const int              max_seq_len,
                   AttentionParam<OpType> param,
                   bool                   use_fused_attention = true);

    // 目前实现的推理根据序列的长度，分为了几类，以适应不同长度区间下不同的 CUDA 优化策略

    //  1. 长于 352 的用 no fused 的方式计算，但是也对最大长度有限制，因为共享内存的原因：
    //      要求 num_heads*seq_len*sizeof(float) 不能超过 64*1024
    //      在 sd 中， num_heads=8, sizeof(float)=4, 那么 seq_len 最大等于 64*1024/8/4 = 2048
    //  2. 长度大于 80，小于 352：
    //      用 attention_fused_long.cu 中实现的长序列下的 fused attention 算法，利用 tensorcore 计算
    //  3. 长度短于 80:
    //      使用用attention_fused_short.cu 中实现的短序列下的 fused attention 算法，利用 tensorcore 计算
    void infer(AttentionInferParam infer_param);
};

}  // namespace lyradiff