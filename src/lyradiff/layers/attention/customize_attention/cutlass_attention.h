
#pragma once
#include <iostream>

#include "attention.h"
#include "cutlass/contrib/args_pack_def.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/device_memory.h"

namespace lyradiff {
namespace cutlass_ops {

// cutlass attention 只支持 fp16 类型
using DataType_ = __half;

template<OperationType OpType>
class CutlassAttention: public Attention<OpType> {
public:
    using Base      = Attention<OpType>;
    using ElementA  = cutlass::half_t;
    using ElementB0 = cutlass::half_t;
    using ElementB1 = cutlass::half_t;
    using ElementC0 = cutlass::half_t;
    using ElementC1 = cutlass::half_t;
    using ElementD  = cutlass::half_t;

protected:
    static constexpr int kMaxThreadblockNumInRow = 32;
    int                  arch_;
    int                  multi_processor_count_;
    using AttentionInferParam = typename Base::AttentionInferParam;

    struct BufSizes {
        unsigned long long input_tensor_size;
        unsigned long long qk_output;
        unsigned long long partial_softmax;
        unsigned long long softmax_reduced;
        unsigned long long total() const
        {
            return input_tensor_size * 3 + qk_output + partial_softmax + softmax_reduced;
        }
    } buf_sizes_;

public:
    CutlassAttention(const int  max_batch_size,
                     const int  head_num,
                     const int  size_per_head,
                     const int  max_seq_len,
                     const bool use_fused_attention = true):
        Base(max_batch_size, head_num, size_per_head, max_seq_len, use_fused_attention)
    {
        static const int arch = []() {
            int major, minor;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
            return major * 10 + minor;
        }();
        arch_ = arch;
        if (arch_ <= 70) {
            multi_processor_count_ = 80;
        }
        else if (arch_ == 75) {
            multi_processor_count_ = 40;
        }
        else if (arch_ >= 80) {
            multi_processor_count_ = 108;
        }
        else {
            printf("[ERROR][BT] unsupported arch: %d!\n", arch_);
            exit(-1);
        }

        // member variables init
        if ((this->max_seq_len_ > 1024) || (this->size_per_head_ != 64)) {
            printf("[ERROR][BT] max_seq_len %d or size_per_head %d is not supported!\n",
                   this->max_seq_len_,
                   this->size_per_head_);
            exit(-1);
        }
        if (this->use_fused_attention_) {
            buf_sizes_.input_tensor_size = 0;
            buf_sizes_.qk_output         = 0;
            buf_sizes_.partial_softmax   = 0;
            buf_sizes_.softmax_reduced   = 0;
        }
        else {
            buf_sizes_.input_tensor_size =
                make_align(1ULL * this->max_batch_size_ * this->head_num_ * this->max_seq_len_ * this->size_per_head_)
                * sizeof(DataType_);
            buf_sizes_.qk_output =
                make_align(1ULL * this->max_batch_size_ * this->head_num_ * this->max_seq_len_ * this->max_seq_len_)
                * sizeof(DataType_);
            buf_sizes_.partial_softmax = make_align(1ULL * this->max_batch_size_ * this->max_seq_len_ * this->head_num_
                                                    * kMaxThreadblockNumInRow)
                                         * sizeof(float) * 2;
            buf_sizes_.softmax_reduced =
                make_align(1ULL * this->max_batch_size_ * this->max_seq_len_ * this->head_num_) * sizeof(float) * 2;
        }
    }

    ~CutlassAttention() override {}

    static unsigned long long make_align(unsigned long long val)
    {
        return ((val + 15) >> 4) << 4;
    }

    unsigned long long cal_bufsize() const override
    {
        return buf_sizes_.total();
    }

    static bool check_seqlen_supported(int seqlen)
    {
        // only suppport unfused attention
        return !(seqlen % 8) && seqlen <= 1024 && seqlen > 384;
    }

    void infer(AttentionInferParam infer_param) override;

protected:
    template<typename CutlassAttentionCore>
    void infer_impl(AttentionInferParam* infer_param);

    template<int SeqLen, int SizePerHead>
    void do_infer(AttentionInferParam* infer_param);
};

}  // namespace cutlass_ops
}  // namespace lyradiff