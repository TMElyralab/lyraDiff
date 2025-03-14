#include <mma.h>

#include "attention.h"
#include "src/lyradiff/common.h"
#include "src/lyradiff/reduce.h"

namespace lyradiff {

#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GPU ARCH SM>70 基础上才支持 TensorCore
#define __CUDA_ARCH__ 800

// 利用 TensorCore WMMA（Warp Matrix Multiply and Accumulate） 进行 Attention 中的 GEMM 计算
// 该 kernel 的一个 TensorCore 同时处理的数据为 16*16 的矩阵块
template<const int max_seq_len, const int size_per_head>  // __launch_bounds__(512,4)//THREADS_PER_BLOCK
__global__ void wmma_attention_kernel_16(const half2*  qkv,
                                         const half2*  qkv_bias,
                                         const __half* attention_mask,
                                         __half*       attention_output,
                                         const int     seq_len,
                                         const float   scale)
{
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
    __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
    __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

    const int warpNums         = (blockDim.x >> 5);
    const int warpId           = (threadIdx.x >> 5);
    const int warp_tid         = getLaneId();
    const int half_hidden_dim  = gridDim.x * (size_per_head / 2);
    const int batch_seq_offset = blockIdx.y * seq_len;
    const int from_size        = max_seq_len / 16;
    const int to_size          = max_seq_len / 16;

    const int quart_warpId        = threadIdx.x >> 3;
    const int quart_warp_tid      = threadIdx.x & 0x7;
    const int quart_thread_offset = blockIdx.x * (size_per_head / 2) + quart_warp_tid;

    // loading Query & Key
    half2 q_bias = __ldg(&qkv_bias[quart_thread_offset]);
    half2 k_bias = __ldg(&qkv_bias[quart_thread_offset + half_hidden_dim]);
    for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
        int pos                        = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
        int offset                     = seq_id * (size_per_head + SKEW_HALF) + (quart_warp_tid << 1);
        *(__half2*)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
        *(__half2*)(*s_kv + offset)    = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
    }

    __syncthreads();

    if (warpId < from_size * to_size) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>               QK_mat;
        wmma::fill_fragment(QK_mat, 0.0f);
        // 左移 4 位，16的偏移量
        const int warp_from_offset = (warpId / to_size) << 4;
        const int warp_to_offset   = (warpId % to_size) << 4;

#pragma unroll
        for (int k = 0; k < 1; k++) {
            wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
            wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
            wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
        }
        wmma::store_matrix_sync(
            s_logits[warp_from_offset] + warp_to_offset, QK_mat, max_seq_len + SKEW_HALF, wmma::mem_row_major);
    }

    __syncthreads();

    // softmax
    for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
        float max_val = -1e20f;

        const int n = (max_seq_len + 31) / 32;
        float     logits[n];
        int       to_id[n];

#pragma unroll
        for (int i = 0; i < n; i++) {
            to_id[i]  = warp_tid + (i << 5);
            logits[i] = -1e20f;

            if (to_id[i] < seq_len) {
                float mask = (float)__ldg(&attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]]);
                mask       = (1.0f - mask) * (-10000.0f);

                logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
            }
            max_val = max(max_val, logits[i]);
        }
        max_val = warpReduceMax(max_val);

        float sum_val = 0.0f;
#pragma unroll
        for (int i = 0; i < n; i++) {
            logits[i] = __expf(logits[i] - max_val);
            sum_val += logits[i];
        }
        sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
        for (int i = 0; i < n; i++)
            if (to_id[i] < max_seq_len)
                s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);
    }

    // 加载 V 值并同时做 bias 加法的融合
    half2 v_bias = __ldg(&qkv_bias[quart_thread_offset + half_hidden_dim * 2]);
    for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
        int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
        ((__half2*)(s_kv[seq_id]))[quart_warp_tid] = __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
    }

    // K dim clear 0
    for (int seq_id = seq_len + quart_warpId; seq_id < max_seq_len; seq_id += warpNums * 4)
        ((float*)(s_kv[seq_id]))[quart_warp_tid] = 0.0f;
    __syncthreads();

    // TensorCore wmma 做 QK softmax 结果 * V
    if (warpId < from_size) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>               QKV_mat;
        wmma::fill_fragment(QKV_mat, 0.0f);
        const int warp_from_offset = (warpId) << 4;
        const int warp_to_offset   = 0;

#pragma unroll
        for (int k = 0; k < to_size; k++) {
            wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K, max_seq_len + SKEW_HALF);
            wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
            wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
        }
        wmma::store_matrix_sync(
            s_query[warp_from_offset] + warp_to_offset, QKV_mat, size_per_head + SKEW_HALF, wmma::mem_row_major);
    }
    __syncthreads();

    for (int from_id = quart_warpId; from_id < seq_len; from_id += warpNums * 4) {
        int pos                             = (batch_seq_offset + from_id) * half_hidden_dim + quart_thread_offset;
        ((__half2*)(attention_output))[pos] = ((__half2*)(s_query[from_id]))[quart_warp_tid];
    }
#endif
}

template<const int max_seq_len, const int size_per_head>
__global__ void wmma_attention_kernel(const half2*  qkv,
                                      const half2*  qkv_bias,
                                      const __half* attention_mask,
                                      __half*       attention_output,
                                      const int     seq_len,
                                      const float   scale)
{
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
    __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
    __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

    const int warpNums         = (blockDim.x >> 5);
    const int warpId           = (threadIdx.x >> 5);
    const int warp_tid         = getLaneId();
    const int half_hidden_dim  = gridDim.x * (size_per_head / 2);
    const int thread_offset    = blockIdx.x * (size_per_head / 2) + warp_tid;
    const int batch_seq_offset = blockIdx.y * seq_len;
    const int from_size        = max_seq_len / 16;
    const int to_size          = max_seq_len / 16;

    // loading Query & Key
    half2 q_bias = __ldg(&qkv_bias[thread_offset]);
    half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
    for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
        int pos                        = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
        int offset                     = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
        *(__half2*)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
        *(__half2*)(*s_kv + offset)    = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
    }
    __syncthreads();

    if (warpId < from_size * to_size) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>               QK_mat;
        wmma::fill_fragment(QK_mat, 0.0f);
        const int warp_from_offset = (warpId / to_size) << 4;
        const int warp_to_offset   = (warpId % to_size) << 4;

#pragma unroll
        for (int k = 0; k < 4; k++) {
            wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
            wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
            wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
        }
        wmma::store_matrix_sync(
            s_logits[warp_from_offset] + warp_to_offset, QK_mat, max_seq_len + SKEW_HALF, wmma::mem_row_major);
    }
    __syncthreads();

    // softmax
    half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
    for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
        const int n = (max_seq_len + 31) / 32;
        float     logits[n];
        int       to_id[n];

        float max_val = -1e20f;
#pragma unroll
        for (int i = 0; i < n; i++) {
            to_id[i]  = warp_tid + (i << 5);
            logits[i] = -1e20f;

            if (to_id[i] < seq_len) {
                float mask = (float)__ldg(&attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]]);
                mask       = (1.0f - mask) * (-10000.0f);
                logits[i]  = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
            }
            max_val = max(max_val, logits[i]);
        }
        max_val = warpReduceMax(max_val);

        float sum_val = 0.0f;
#pragma unroll
        for (int i = 0; i < n; i++) {
            logits[i] = __expf(logits[i] - max_val);
            sum_val += logits[i];
        }
        sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
        for (int i = 0; i < n; i++)
            if (to_id[i] < max_seq_len)
                s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);

        // loading Value
        int pos                               = (batch_seq_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
        ((__half2*)(s_kv[from_id]))[warp_tid] = __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
    }

    // K dim clear 0
    for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
        ((float*)(s_kv[seq_id]))[warp_tid] = 0.0f;
    __syncthreads();

    //* V
    if (warpId < (from_size << 2)) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>               QKV_mat;
        wmma::fill_fragment(QKV_mat, 0.0f);
        const int warp_from_offset = (warpId >> 2) << 4;
        const int warp_to_offset   = (warpId & 0x3) * WMMA_K;

#pragma unroll
        for (int k = 0; k < to_size; k++) {
            wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K, max_seq_len + SKEW_HALF);
            wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
            wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
        }
        wmma::store_matrix_sync(
            s_query[warp_from_offset] + warp_to_offset, QKV_mat, size_per_head + SKEW_HALF, wmma::mem_row_major);
    }
    __syncthreads();

    for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
        int pos                             = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
        ((__half2*)(attention_output))[pos] = ((__half2*)(s_query[from_id]))[warp_tid];
    }
#endif
}

#define WMMA_ATTENTION_16(SEQ_LEN, SIZE_PER_HEAD)                                                                      \
    wmma_attention_kernel_16<SEQ_LEN, SIZE_PER_HEAD><<<grid, block, 0, infer_param.stream>>>(                          \
        qkv_ptr, qkv_bias_ptr, (__half*)atten_mask, (__half*)attention_output, seq_len, scale)

#define WMMA_ATTENTION(SEQ_LEN, SIZE_PER_HEAD)                                                                         \
    wmma_attention_kernel<SEQ_LEN, SIZE_PER_HEAD><<<grid, block, 0, infer_param.stream>>>(                             \
        qkv_ptr, qkv_bias_ptr, (__half*)atten_mask, (__half*)attention_output, seq_len, scale)

// Attention 的 fused 推理函数实现
template<OperationType OpType>
void Attention<OpType>::fused_infer(AttentionInferParam infer_param)
{
    const DataType_* atten_mask       = infer_param.atten_mask;
    DataType_*       attention_output = infer_param.attention_output;
    const int        batch_size       = infer_param.batch_size;
    const int        seq_len          = infer_param.seq_len;

    dim3 grid(head_num_, batch_size), block;

    if (OpType == OperationType::HALF) {
        const half2* qkv_ptr      = (const half2*)infer_param.qkv;
        const half2* qkv_bias_ptr = (const half2*)param_.attr_bias_QKV;
        float        scale        = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

        // block 为一维，线程数为 seq/16 和 max(seq/16, size_per_head/16) 的乘积再 乘 32
        block.x = 32 * ((seq_len + 15) / 16) * max(((seq_len + 15) / 16), size_per_head_ / 16);

        // 因为一个 block 中最大线程数为 1024，那么 反推 seq_len 和 size_perhead 的受  1024/32=32 的限制
        // seq/16 和 max(seq/16, size_per_head/16) 的乘积不能超过 32
        if (size_per_head_ == 64) {
            if (seq_len <= 16)
                WMMA_ATTENTION(16, 64);
            else if (seq_len <= 32)
                WMMA_ATTENTION(32, 64);  // 32/16 * 64/16 == 8
            else if (seq_len <= 48)
                WMMA_ATTENTION(48, 64);  // 48/16 * 64/16 == 12
            else if (seq_len <= 64)
                WMMA_ATTENTION(64, 64);  // 64/16 * 64/16 == 16
            else if (seq_len <= 80)
                WMMA_ATTENTION(80, 64);  // 80/16 * 64/16 == 20
            // else if (seq_len <= 128)
            //     WMMA_ATTENTION(128, 64);  // 128/16 * 64/16 == 32  128 时共享数据空间在 A100 上发现已经不够了
        }
        else if (size_per_head_ == 16) {
            if (seq_len <= 48)
                WMMA_ATTENTION_16(48, 16);
        }
    }
}

template void Attention<OperationType::FP32>::fused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_infer(AttentionInferParam infer_param);

}  // namespace lyradiff