#pragma once
#include "src/lyradiff/common.h"

namespace lyradiff {

// Attention 构造参数模板类
template<OperationType OpType>
class AttentionParam {
private:
    using Traits_   = Traits<OpType>;
    using DataType_ = typename Traits_::DataType;

public:
    const DataType_* attr_bias_QKV;  // [hidden_dim * 3]
    float            tao;

    // 用于指定 cublas 矩阵乘法的算法
    int cublas_Algo[2];

    AttentionParam()
    {
        attr_bias_QKV = nullptr;
        tao           = 1.0f;

        cublas_Algo[0] = cublas_Algo[1] = Traits_::algo;
    }
};

// Attention 推理时参数模板类
template<typename T>
struct AttentionInferParam {
    void*          buf;
    T*             attention_output;          // [batch_size, seq_len, hidden_dim]
    const T*       qkv;                       // [batch_size, seq_len, hidden_dim * 3]
    const T*       atten_mask;                // [batch_size, seq_len, seq_len], [1, 0]
    const T*       attention_bias = nullptr;  // [head_num, seq_len, seq_len]
    int            batch_size     = 0;
    int            seq_len        = 0;
    cublasHandle_t cublas_handle  = nullptr;
    cudaStream_t   stream         = nullptr;
};

// Attention Layer 模板类
template<OperationType OpType>
class Attention {
protected:
    typedef Traits<OpType>             Traits_;
    typedef typename Traits_::DataType DataType_;
    const int                          max_batch_size_, max_seq_len_, head_num_, size_per_head_;
    AttentionParam<OpType>             param_;

    using AttentionInferParam = struct AttentionInferParam<DataType_>;

    bool use_fused_attention_;

    bool transformer_variety_fuse_flag_;

public:
    Attention(const int  max_batch_size,
              const int  head_num,
              const int  size_per_head,
              const int  max_seq_len,
              const bool use_fused_attention = true):
        max_batch_size_(max_batch_size),
        max_seq_len_(max_seq_len),
        head_num_(head_num),
        size_per_head_(size_per_head),
        use_fused_attention_(use_fused_attention)
    {
        // fused attention 仅作用在 fp16 的情况下
        if (OpType == OperationType::FP32)
            use_fused_attention_ = false;

        transformer_variety_fuse_flag_ = use_fused_attention_;

        if (transformer_variety_fuse_flag_) {
            if (size_per_head != 64)
                transformer_variety_fuse_flag_ = false;

            if (max_seq_len_ > 256)
                transformer_variety_fuse_flag_ = false;
        }
    }

    void initialize(AttentionParam<OpType> param)
    {
        param_ = param;
    }

    virtual unsigned long long cal_bufsize() const
    {
        if (use_fused_attention_)
            return 0;
        else {
            unsigned long long input_tensor_size = max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
            // 加 15, 右移 4 位，再左移 4 位，让内存对齐 16 的倍数
            unsigned long long qk_buf_size = ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >> 4)
                                             << 4;  // for memory alignment
            unsigned long long inner_buf_size = input_tensor_size * 4 + qk_buf_size;
            unsigned long long total_buf_size = inner_buf_size * sizeof(DataType_);
            return total_buf_size;
        }
    }

    virtual void infer(AttentionInferParam infer_param)
    {
        if (use_fused_attention_) {
            if (infer_param.seq_len <= 128) {
                fused_infer(infer_param);
            }
            else {
                fused_long_infer(infer_param);
            }
        }
        else
            nofused_infer(infer_param);
    }

    // 长于 352 的用 no fused
    void nofused_infer(AttentionInferParam infer_param);

    // 长度短于 80， 用 attention_fused_short.cu 中的实现
    void fused_infer(AttentionInferParam infer_param);
    // 长度大于 80，小于 352 内，用 attention_fused_long.cu 中的实现
    void fused_long_infer(AttentionInferParam infer_param);

    virtual ~Attention() {}
};

}  // namespace lyradiff