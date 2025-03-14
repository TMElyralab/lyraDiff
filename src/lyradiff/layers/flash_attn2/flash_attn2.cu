
#include "flash_attn2.h"
#include <cmath>

namespace lyradiff {
namespace flash_attn2 {

template<typename T>
void FlashAttention2Layer<T>::allocateBuffer()
{
}

template<typename T>
void FlashAttention2Layer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t softmax_lse_size = batch_size * head_num_ * seq_len;

    softmax_lse_ = (float*)allocator_->reMalloc(softmax_lse_, sizeof(float) * softmax_lse_size, true);

    is_allocate_buffer_ = true;
}

template<typename T>
void FlashAttention2Layer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&softmax_lse_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
FlashAttention2Layer<T>::~FlashAttention2Layer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
    softmax_lse_ = nullptr;
}

template<typename T>
FlashAttention2Layer<T>::FlashAttention2Layer(const size_t     head_num,
                                              const size_t     hidden_units,
                                              const int32_t    sm,
                                              cudaStream_t     stream,
                                              cublasMMWrapper* cublas_wrapper,
                                              IAllocator*      allocator,
                                              bool             is_free_buffer_after_forward,
                                              const bool       is_causal):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    head_num_(head_num),
    hidden_units_(hidden_units),
    sm_(sm),
    is_causal_(is_causal)
{
    softmax_scale_ = 1.F / sqrtf(hidden_units);
}

template<typename T>
void FlashAttention2Layer<T>::forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors)
{
    TensorMap input_tensor(
        {{"q_buf", input_tensors->at(0)}, {"k_buf", input_tensors->at(1)}, {"v_buf", input_tensors->at(2)}});
    TensorMap output_tensor({{"attn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor);
}

template<typename T>
void FlashAttention2Layer<T>::forward(TensorMap* output_tensor, TensorMap* input_tensor)
{
    // input_tensor: 必须有 qkv_buf 键代表输入
    //      q_buf 应该是 [B, SeqlenQ, NumHeadQ, PerSizeHead] 拍平的显存数据
    //      k_buf 应该是 [B, SeqlenK, NumHeadK, PerSizeHead] 拍平的显存数据
    //      v_buf 应该是 [B, SeqlenV, NumHeadV, PerSizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLenQ, NumHeadQ, PerSizeHead] 拍平的显存数据

    LYRA_CHECK(input_tensor->size() == 3);
    LYRA_CHECK(output_tensor->size() == 1);

    T* attn_output = output_tensor->at("attn_output").getPtr<T>();

    const T* q = input_tensor->getPtr<const T>("q_buf", nullptr);
    const T* k = input_tensor->getPtr<const T>("k_buf", nullptr);
    const T* v = input_tensor->getPtr<const T>("v_buf", nullptr);

    size_t batch_size = input_tensor->at("q_buf").shape[0];
    size_t seq_len_q  = input_tensor->at("q_buf").shape[1];
    size_t nheads_q   = input_tensor->at("q_buf").shape[2];
    size_t head_size  = input_tensor->at("q_buf").shape[3];

    size_t seq_len_k = input_tensor->at("k_buf").shape[1];
    size_t nheads_k  = input_tensor->at("k_buf").shape[2];

    size_t seq_len_v = input_tensor->at("v_buf").shape[1];
    size_t nheads_v  = input_tensor->at("v_buf").shape[2];

    allocateBuffer(batch_size, seq_len_q);


    // sync_check_cuda_error();

    invokeFlashAttn2Fwd(attn_output,
                        softmax_lse_,
                        q,
                        k,
                        v,
                        {uint32_t(batch_size), uint32_t(seq_len_q), uint32_t(nheads_q), uint32_t(head_size)},
                        {uint32_t(batch_size), uint32_t(seq_len_k), uint32_t(nheads_k), uint32_t(head_size)},
                        {uint32_t(batch_size), uint32_t(seq_len_v), uint32_t(nheads_v), uint32_t(head_size)},
                        {uint32_t(batch_size), uint32_t(seq_len_q), uint32_t(nheads_q), uint32_t(head_size)},
                        softmax_scale_,
                        is_causal_,
                        stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    // sync_check_cuda_error();
}

template class FlashAttention2Layer<half>;
template class FlashAttention2Layer<float>;
}  // namespace flash_attn2
}  // namespace lyradiff
