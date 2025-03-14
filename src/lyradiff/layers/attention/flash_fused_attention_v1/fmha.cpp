
#include "fmha.h"
#include <cmath>

namespace lyradiff {
namespace flash_attn {

int32_t runFMHFAKernel(void const*                               devQKV,
                       void*                                     cuSeqlens,
                       void*                                     devOutput,
                       size_t                                    total,
                       int32_t                                   sm,
                       FusedMultiHeadFlashAttentionKernel const* kernels,
                       int32_t                                   b,
                       int32_t                                   h,
                       int32_t                                   d,
                       int32_t                                   s,
                       cudaStream_t                              stream)
{
    Fused_multihead_flash_attention_params_v2 params = getMHFAParams(/* data_type */ DATA_TYPE_FP16,
                                                                     /* acc_type */ DATA_TYPE_FP16,
                                                                     b,
                                                                     s,
                                                                     h,
                                                                     d,
                                                                     total,
                                                                     devQKV,
                                                                     cuSeqlens,
                                                                     devOutput,
                                                                     /* p_d */ nullptr,
                                                                     /* s_d */ nullptr,
                                                                     /* scale_bmm1 */ 1.F / sqrtf(d),
                                                                     /* scale_softmax */ 1.F,
                                                                     /* scale_bmm2 */ 1.F,
                                                                     /* interleaved */ false,
                                                                     /* ignore_b1opt */ false,
                                                                     /* force_unroll */ true,
                                                                     /* use_int8_scale_max  */ false);

    kernels->run(params, stream);
    return 0;
}

template<typename T>
void FusedFlashAttentionLayerV1<T>::allocateBuffer()
{
}

template<typename T>
void FusedFlashAttentionLayerV1<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    cu_seq_lens_ = (int32_t*)allocator_->reMalloc(cu_seq_lens_, sizeof(int32_t) * (batch_size + 1), false);

    if (pre_seq_len_ != seq_len || pre_batch_size_ != batch_size) {
        std::vector<int32_t> cu_seq_lens_host(batch_size + 1, 0);
        // Compute the prefix sum of the seqlen
        for (int32_t it = 0; it < batch_size; it++) {
            cu_seq_lens_host[it + 1] = cu_seq_lens_host[it] + seq_len;
        }
        cudaMemcpyAsync(cu_seq_lens_,
                        cu_seq_lens_host.data(),
                        sizeof(int32_t) * cu_seq_lens_host.size(),
                        cudaMemcpyHostToDevice,
                        stream_);

        pre_batch_size_ = batch_size;
        pre_seq_len_    = seq_len;
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void FusedFlashAttentionLayerV1<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&cu_seq_lens_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
FusedFlashAttentionLayerV1<T>::~FusedFlashAttentionLayerV1()
{
    cublas_wrapper_ = nullptr;
    kernels_        = nullptr;
    freeBuffer();
}

template<typename T>
FusedFlashAttentionLayerV1<T>::FusedFlashAttentionLayerV1(size_t           head_num,
                                                          size_t           hidden_units,
                                                          int32_t          sm,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    head_num_(head_num),
    hidden_units_(hidden_units),
    sm_(sm)
{
    if (std::is_same<T, half>::value) {
        kernels_ = getFMHAFlashCubinKernels(MHFADataType::DATA_TYPE_FP16, sm_);
    }
    else if (std::is_same<T, float>::value) {
        kernels_ = getFMHAFlashCubinKernels(MHFADataType::DATA_TYPE_FP32, sm_);
    }
    else {
        throw "the typename must be half or float";
    }
}

template<typename T>
void FusedFlashAttentionLayerV1<T>::forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors)
{
    TensorMap input_tensor({{"qkv_buf", input_tensors->at(0)}});
    TensorMap output_tensor({{"attn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor);
}

template<typename T>
void FusedFlashAttentionLayerV1<T>::forward(TensorMap* output_tensor, TensorMap* input_tensor)
{
    // input_tensor: 必须有 qkv_buf 键代表输入
    //      qkv_buf 应该是 [B, Seqlen, NumHead, 3, PerSizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLen, NumHead, PerSizeHead] 拍平的显存数据

    LYRA_CHECK(input_tensor->size() == 1);
    LYRA_CHECK(output_tensor->size() == 1);

    const T* qkv_buf     = input_tensor->getPtr<const T>("qkv_buf", nullptr);
    T*       attn_output = output_tensor->at("attn_output").getPtr<T>();
    size_t   batch_size  = input_tensor->at("qkv_buf").shape[0];
    size_t   seq_len     = input_tensor->at("qkv_buf").shape[1];

    allocateBuffer(batch_size, seq_len);

    // sync_check_cuda_error();
    size_t total = batch_size * seq_len;
    runFMHFAKernel(qkv_buf,
                   cu_seq_lens_,
                   attn_output,
                   total,
                   sm_,
                   kernels_,
                   batch_size,
                   head_num_,
                   hidden_units_,
                   seq_len,
                   stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    // sync_check_cuda_error();
}

template class FusedFlashAttentionLayerV1<half>;
template class FusedFlashAttentionLayerV1<float>;
}  // namespace flash_attn
}  // namespace lyradiff
