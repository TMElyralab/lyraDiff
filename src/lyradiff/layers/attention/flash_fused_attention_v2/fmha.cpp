
#include "fmha.h"
#include <cmath>

namespace lyradiff {
namespace flash_attn {

template<typename T>
void FusedFlashAttentionLayerV2<T>::allocateBuffer()
{
}

template<typename T>
void FusedFlashAttentionLayerV2<T>::allocateBuffer(size_t batch_size, size_t seq_len)
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
void FusedFlashAttentionLayerV2<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&cu_seq_lens_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
FusedFlashAttentionLayerV2<T>::~FusedFlashAttentionLayerV2()
{
    cublas_wrapper_ = nullptr;
    delete runner_;
    freeBuffer();
}

template<typename T>
FusedFlashAttentionLayerV2<T>::FusedFlashAttentionLayerV2(size_t           head_num,
                                                          size_t           hidden_units,
                                                          size_t           kv_head_num,
                                                          float            q_scaling,
                                                          const bool       force_fp32_acc,
                                                          const bool       is_s_padded,
                                                          const bool       causal_mask,
                                                          int32_t          sm,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward,
                                                          const bool       has_alibi,
                                                          const bool       scale_alibi,
                                                          const int        tp_size,
                                                          const int        tp_rank):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    head_num_(head_num),
    hidden_units_(hidden_units),
    q_scaling_(q_scaling),
    kv_head_num_(kv_head_num),
    causal_mask_(causal_mask),
    sm_(sm),
    has_alibi_(has_alibi),
    scale_alibi_(scale_alibi),
    tp_size_(tp_size_),
    tp_rank_(tp_rank)

{
    // std::cout << "head_num: " << head_num << std::endl;
    // std::cout << "hidden_units: " << hidden_units << std::endl;
    // std::cout << "q_scaling: " << q_scaling << std::endl;
    // 1. 创建 runner 对象
    if (std::is_same<T, half>::value) {
        runner_ = new FusedMHARunnerV2(DATA_TYPE_FP16, head_num, hidden_units, q_scaling);
    }
    else if (std::is_same<T, float>::value) {
        runner_ = new FusedMHARunnerV2(DATA_TYPE_FP32, head_num, hidden_units, q_scaling);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        runner_ = new FusedMHARunnerV2(DATA_TYPE_BF16, head_num, hidden_units, q_scaling);
    }
#endif
    else {
        throw std::runtime_error("unsupported type");
    }
    // 2. 检查 fmha 是否在当前的机器和问题参数上可以运行
    if (runner_->fmha_supported()) {
        // 3. 设置 flag
        runner_->setup_flags(force_fp32_acc, is_s_padded, causal_mask, kv_head_num);
    }
    else {
        throw std::runtime_error("Your config is not suppported by flash attention");
    }
}

template<typename T>
void FusedFlashAttentionLayerV2<T>::forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors)
{
    TensorMap input_tensor({{"qkv_buf", input_tensors->at(0)}});
    TensorMap output_tensor({{"attn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor);
}

template<typename T>
void FusedFlashAttentionLayerV2<T>::forward(TensorMap* output_tensor, TensorMap* input_tensor)
{
    // input_tensor: 必须有 qkv_buf 键代表输入
    //      qkv_buf 应该是 [B, Seqlen, 3, NumHead, PerSizeHead] 拍平的显存数据
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
    size_t total_seq = batch_size * seq_len;

    // 设置运行时
    runner_->setup(batch_size, seq_len, total_seq, has_alibi_, scale_alibi_, tp_size_, tp_rank_);

    // 开始计算
    runner_->run(qkv_buf, cu_seq_lens_, attn_output, stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    // sync_check_cuda_error();
}

template class FusedFlashAttentionLayerV2<half>;
template class FusedFlashAttentionLayerV2<float>;
#ifdef ENABLE_BF16
template class FusedFlashAttentionLayerV2<__nv_bfloat16>;
#endif
}  // namespace flash_attn
}  // namespace lyradiff
