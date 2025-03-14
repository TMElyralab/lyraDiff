#include "attention.h"
#include "attention_nofused_utils.h"
#include "src/lyradiff/kernels/basic/gemm.h"
#include "src/lyradiff/kernels/basic/softmax.h"
#include "variety_attention_fused.h"

namespace lyradiff {
template<OperationType OpType>
void Attention<OpType>::nofused_infer(AttentionInferParam infer_param)
{
    void*        buf        = infer_param.buf;
    const int    batch_size = infer_param.batch_size;
    const int    seq_len    = infer_param.seq_len;
    cudaStream_t stream     = infer_param.stream;

    int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;
    // 对齐内存 16 位倍数
    int qk_buf_size = ((batch_size * head_num_ * seq_len * seq_len + 15) >> 4) << 4;

    DataType_* query         = (DataType_*)buf + 0 * input_tensor_size;
    DataType_* key           = (DataType_*)buf + 1 * input_tensor_size;
    DataType_* value         = (DataType_*)buf + 2 * input_tensor_size;
    DataType_* qk_buf        = (DataType_*)buf + 3 * input_tensor_size;
    DataType_* transpose_dst = qk_buf + qk_buf_size;

    int size_per_head_half = (OpType == OperationType::HALF) ? size_per_head_ / 2 : size_per_head_;  // Be careful.

    // [batch_size, seq_len, hidden_dim] ->
    // [head_num, batch_size, seq_len, size_per_head]
    dim3 grid, block;
    grid.x = seq_len, grid.y = batch_size;
    block.x                = head_num_ * (size_per_head_ / 2);  // Process two adjacent values for float/half
    const bool is_roformer = false;
    add_QKV_bias<<<grid, block, 0, stream>>>(infer_param.qkv,
                                             param_.attr_bias_QKV,
                                             query,
                                             key,
                                             value,
                                             batch_size,
                                             seq_len,
                                             head_num_,
                                             size_per_head_ / 2,
                                             is_roformer);
    grid.y = 1;

    DataType_ alpha = (DataType_)(1.0f / sqrtf(size_per_head_ * 1.0f) / param_.tao), beta = (DataType_)0.0f;
    bool      add_qk_buf = false;

    if (transformer_variety_fuse_flag_)
        variety_attention_fused_infer((const __half*)query,
                                      (const __half*)key,
                                      (const __half*)value,
                                      (const __half*)infer_param.atten_mask,
                                      add_qk_buf ? (const __half*)qk_buf : NULL,
                                      (const __half*)infer_param.attention_bias,
                                      (__half*)infer_param.attention_output,
                                      head_num_,
                                      batch_size,
                                      seq_len,
                                      size_per_head_,
                                      (float)alpha,
                                      infer_param.stream,
                                      NULL);
    else {
        cublas_Gemm_Strided_Batched(query,
                                    key,
                                    qk_buf,
                                    seq_len,
                                    size_per_head_,
                                    seq_len,
                                    head_num_ * batch_size,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_T,
                                    alpha,
                                    beta,
                                    infer_param.cublas_handle,
                                    stream,
                                    param_.cublas_Algo[0]);

        // qk_buf 如果 head_num * seq_len * sizeof(float) > 64*1024，共享内存将不够
        // 非 flashattention 中 softmax 的实现
        softmax_kernelLauncher<OpType, DataType_>(
            qk_buf, infer_param.attention_bias, infer_param.atten_mask, batch_size, seq_len, head_num_, stream);

        alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
        cublas_Gemm_Strided_Batched(qk_buf,
                                    value,
                                    transpose_dst,
                                    seq_len,
                                    seq_len,
                                    size_per_head_,
                                    head_num_ * batch_size,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    alpha,
                                    beta,
                                    infer_param.cublas_handle,
                                    stream,
                                    param_.cublas_Algo[1]);

        block.x = size_per_head_half, block.y = head_num_;

        transpose<<<batch_size * seq_len, block, 0, stream>>>(
            transpose_dst, infer_param.attention_output, batch_size, seq_len, head_num_, size_per_head_half);
    }
}

template void Attention<OperationType::FP32>::nofused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::nofused_infer(AttentionInferParam infer_param);
}  // namespace lyradiff