
#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "tmaDescriptor.h"

namespace lyradiff {

////////////////////////////////////////////////////////////////////////////////////////////////////

class MHARunner {
public:
    MHARunner(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    MHARunner() = default;

    virtual ~MHARunner() = default;

    virtual void setup(const int  b,
                       const int  s,
                       const int  total_seqlen,
                       const bool has_alibi   = false,
                       const bool scale_alibi = false,
                       const int  tp_size     = 1,
                       const int  tp_rank     = 0) = 0;

    static bool fmha_supported(const int headSize, const int sm);

    virtual bool fmha_supported() = 0;

    virtual void setup_flags(const bool force_fp32_acc,
                             const bool is_s_padded,
                             const bool causal_mask,
                             const int  num_kv_heads /* MQA or GQA */) = 0;

    virtual void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) = 0;

    virtual bool isValid(int s) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workflow of fmha runner:
// 1. check if FMHA kernels are supported statically.
// 2. construct FMHA runner object.
// 3. setup_flags (used by all kernels).
// 4. setup runtime parameters (used by this specific case).
// 5. run the kernel (with all needed device pointers).

class FusedMHARunnerV2: public MHARunner {
public:
    FusedMHARunnerV2(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    ~FusedMHARunnerV2();  // for pimpl

    void setup(const int  b,
               const int  s,
               const int  total_seqlen,
               const bool has_alibi   = false,
               const bool scale_alibi = false,
               const int  tp_size     = 1,
               const int  tp_rank     = 0) override;

    bool fmha_supported() override;

    void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) override;

    void setup_flags(const bool force_fp32_acc,
                     const bool is_s_padded,
                     const bool causal_mask,
                     const int  num_kv_heads /* MQA or GQA */) override;

    bool isValid(int s) const override;

private:
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

}  // namespace lyradiff
