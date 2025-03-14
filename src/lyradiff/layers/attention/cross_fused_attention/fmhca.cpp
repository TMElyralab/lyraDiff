#include "fmhca.h"

namespace lyradiff {
namespace cross_attn {
int32_t runFMHCAKernel(void const*                               devQ,
                       void const*                               devKV,
                       void*                                     cuSeqlensQ,
                       void*                                     cuSeqlensKV,
                       void*                                     devOutput,
                       int32_t                                   sm,
                       FusedMultiHeadCrossAttentionKernel const* kernels,
                       int32_t                                   b,
                       int32_t                                   h,
                       int32_t                                   d,
                       int32_t                                   seqQ,
                       int32_t                                   seqKV,
                       cudaStream_t                              stream)
{

    // PLUGIN_VALIDATE(sm != kSM_75 || d < 160, "There are no fMHCA kernels for sm75 and d >= 160.");

    // Run kernel.
    Fused_multihead_attention_params_mhca params = getMHCAParams(/* dType */ DATA_TYPE_FP16,
                                                                 /* accType */ DATA_TYPE_FP16,
                                                                 b,
                                                                 seqQ,
                                                                 seqKV,
                                                                 h,
                                                                 d,
                                                                 /* total */ 0,
                                                                 devQ,
                                                                 devKV,
                                                                 cuSeqlensQ,
                                                                 cuSeqlensKV,
                                                                 devOutput,
                                                                 /* devP */ nullptr,
                                                                 /* devS */ nullptr,
                                                                 /* scaleBmm1 */ 1.F / sqrtf(d),
                                                                 /* scaleSoftmax */ 1.F,
                                                                 /* scaleBmm2 */ 1.F,
                                                                 /* interleaved */ false,
                                                                 /* ignoreB1Opt */ false,
                                                                 /* forceUnroll */ true,
                                                                 /* useInt8ScaleMax */ false,
                                                                 /* useTMA */ false);

    kernels->run(params, stream);
    return 0;
}

template<typename T>
void FusedCrossAttentionLayer<T>::allocateBuffer()
{
}

template<typename T>
void FusedCrossAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t q_seq_len, size_t kv_seq_len)
{
    // Batch size 不变就复用之前分配过的显存，发生改变就重新分配
    q_cu_seq_lens_  = (int32_t*)allocator_->reMalloc(q_cu_seq_lens_, sizeof(int32_t) * (batch_size + 1), false);
    kv_cu_seq_lens_ = (int32_t*)allocator_->reMalloc(kv_cu_seq_lens_, sizeof(int32_t) * (batch_size + 1), false);

    if (pre_batch_size_ != batch_size || pre_q_seq_len_ != q_seq_len || pre_kv_seq_len_ != kv_seq_len) {

        std::vector<int32_t> q_cu_seq_lens_host(batch_size + 1, 0);
        std::vector<int32_t> kv_cu_seq_lens_host(batch_size + 1, 0);
        // Compute the prefix sum of the seqlen
        for (int32_t it = 0; it < batch_size; it++) {
            q_cu_seq_lens_host[it + 1]  = q_cu_seq_lens_host[it] + q_seq_len;
            kv_cu_seq_lens_host[it + 1] = kv_cu_seq_lens_host[it] + kv_seq_len;
        }
        cudaMemcpyAsync(q_cu_seq_lens_,
                        q_cu_seq_lens_host.data(),
                        sizeof(int32_t) * q_cu_seq_lens_host.size(),
                        cudaMemcpyHostToDevice,
                        stream_);
        cudaMemcpyAsync(kv_cu_seq_lens_,
                        kv_cu_seq_lens_host.data(),
                        sizeof(int32_t) * kv_cu_seq_lens_host.size(),
                        cudaMemcpyHostToDevice,
                        stream_);

        pre_batch_size_ = batch_size;
        pre_q_seq_len_  = q_seq_len;
        pre_kv_seq_len_ = kv_seq_len;
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void FusedCrossAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&q_cu_seq_lens_));
        allocator_->free((void**)(&kv_cu_seq_lens_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
FusedCrossAttentionLayer<T>::~FusedCrossAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    kernels_        = nullptr;
    freeBuffer();
}

template<typename T>
FusedCrossAttentionLayer<T>::FusedCrossAttentionLayer(size_t           head_num,
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
        kernels_ = getFMHCACubinKernels(MHCADataType::DATA_TYPE_FP16, sm_);
    }
    else if (std::is_same<T, float>::value) {
        kernels_ = getFMHCACubinKernels(MHCADataType::DATA_TYPE_FP32, sm_);
    }
    else {
        throw "the typename must be half or float";
    }
}

template<typename T>
void FusedCrossAttentionLayer<T>::forward(std::vector<Tensor>* output_tensors, std::vector<Tensor>* input_tensors)
{
    TensorMap input_tensor({{"q_buf", input_tensors->at(0)}, {"kv_buf", input_tensors->at(1)}});
    TensorMap output_tensor({{"attn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor);
}

template<typename T>
void FusedCrossAttentionLayer<T>::forward(TensorMap* output_tensor, TensorMap* input_tensor)
{
    // input_tensor: 必须有 q_buf 键 和 kv_buf 代表输入
    //      q_buf 应该是 [B, QSeqlen, NumHead, PerSizeHead] 拍平的显存数据
    //      kv_buf 应该是 [B, KVSeqlen, NumHead, 2, PersizeHead] 拍平的显存数据
    // output_tensor: 输出 attn_output
    //      attn_output 是 [B, SeqLen, NumHead, PerSizeHead] 拍平的显存数据

    LYRA_CHECK(input_tensor->size() == 2);
    LYRA_CHECK(output_tensor->size() == 1);

    const T* q_buf       = input_tensor->getPtr<const T>("q_buf", nullptr);
    const T* kv_buf      = input_tensor->getPtr<const T>("kv_buf", nullptr);
    T*       attn_output = output_tensor->at("attn_output").getPtr<T>();

    size_t batch_size = input_tensor->at("q_buf").shape[0];
    size_t q_seq_len  = input_tensor->at("q_buf").shape[1];
    size_t kv_seq_len = input_tensor->at("kv_buf").shape[1];


    // sync_check_cuda_error();

    allocateBuffer(batch_size, q_seq_len, kv_seq_len);

    constexpr int32_t kv_seq_len_padded = 128;
    runFMHCAKernel(q_buf,
                   kv_buf,
                   q_cu_seq_lens_,
                   kv_cu_seq_lens_,
                   attn_output,
                   sm_,
                   kernels_,
                   batch_size,
                   head_num_,
                   hidden_units_,
                   q_seq_len,
                   kv_seq_len_padded,
                   stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    // sync_check_cuda_error();
}

template class FusedCrossAttentionLayer<half>;
template class FusedCrossAttentionLayer<float>;
}  // namespace cross_attn
}  // namespace lyradiff
