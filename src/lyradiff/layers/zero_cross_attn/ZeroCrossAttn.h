#pragma once

#include "ZeroCrossAttnWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>
// #include "src/lyradiff/layers/attention/cross_fused_attention/fmhca.h"
#include "src/lyradiff/layers/flash_attn2/flash_attn2.h"

namespace lyradiff {

template<typename T>
class ZeroCrossAttn: public BaseLayer {
private:
    // params for 2 Identical ResNets
    size_t query_dim_;
    size_t context_dim_;
    size_t heads_;
    size_t dim_head_ = 64;
    size_t inner_dim_;
    size_t norm_num_groups_ = 32;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, size_t context_seq_len);

public:
    // Cross Attention part
    lyradiff::flash_attn2::FlashAttention2Layer<T>* flash_attn2_layer;

    T*      attn_q_buf_       = nullptr;
    T*      attn_kv_buf_      = nullptr;
    T*      attn_kv_buf2_     = nullptr;
    T*      context_buf_      = nullptr;
    T*      hidden_state_buf_ = nullptr;
    T*      attn_output_buf_  = nullptr;
    double* norm_cache_buf_   = nullptr;

    ZeroCrossAttn(size_t           query_dim,
                  size_t           context_dim,
                  cudaStream_t     stream,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  const bool       is_free_buffer_after_forward,
                  const bool       sparse);

    ZeroCrossAttn(ZeroCrossAttn<T> const& other);

    virtual ~ZeroCrossAttn();

    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors, const ZeroCrossAttnWeight<T>* weights, const float control_scale=0.0);
};

}  // namespace lyradiff
