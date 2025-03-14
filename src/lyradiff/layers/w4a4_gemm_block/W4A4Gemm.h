#pragma once

#include "W4A4GemmWeight.h"
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {

template<typename T>
class W4A4Gemm: public BaseLayer {
private:
    float*    lora_down_res_buffer;
    T*        lora_down_res_buffer2;
    float*    input_scale_buffer;
    uint32_t* quantized_input_buffer;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t m);

protected:
    size_t d_out_;
    size_t d_in_;
    size_t lora_rank_  = 32;
    size_t group_size_ = 64;
    size_t group_num_  = 0;
    bool   has_bias_   = true;

public:
    W4A4Gemm(size_t              d_out,
             size_t              d_in,
             size_t              lora_rank,
             bool                has_bias,
             size_t              group_size,
             cudaStream_t        stream,
             cublasMMWrapper*    cublas_wrapper,
             IAllocator*         allocator,
             const bool          is_free_buffer_after_forward,
             const bool          sparse,
             const LyraQuantType quant_level = LyraQuantType::NONE);

    W4A4Gemm(W4A4Gemm<T> const& other);

    virtual ~W4A4Gemm();

    virtual void
    forward(const TensorMap* output_tensors, const TensorMap* input_tensors, const W4A4GemmWeight<T>* weights);
};

}  // namespace lyradiff
