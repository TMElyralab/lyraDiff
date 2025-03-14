#pragma once
#include "src/lyradiff/common.h"
#include "src/lyradiff/reduce.h"

namespace lyradiff {
template<typename T>
__global__ void softmax_kernel_warp(
    T* qk_buf, const T* atten_bias, const T* atten_mask, const int batch_size, const int head_num, const int seq_len)
{
    int word_id   = blockIdx.x;
    int batch_id  = word_id / seq_len;
    int seq_id    = word_id % seq_len;
    int warp_tid  = threadIdx.x;
    int head_id   = threadIdx.y;
    int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

    extern __shared__ float shmem[];
    float*                  s_row_qk = (float*)shmem + head_id * seq_len;

    float max_v = -1e20f;
    for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
        float qk = (float)qk_buf[qk_offset + col_id];
        if (atten_bias)
            qk += (float)atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id];
        float mask_val   = (1.0f - (float)atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]) * -10000.0f;
        float tmp        = qk + mask_val;
        s_row_qk[col_id] = tmp;
        max_v            = tmp > max_v ? tmp : max_v;
    }
    max_v = warpReduceMax<float>(max_v);

    float exp_sum = 0.0f;
    for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
        float qk         = __expf(s_row_qk[col_id] - max_v);
        s_row_qk[col_id] = qk;
        exp_sum += qk;
    }
    exp_sum = warpReduceSum<float>(exp_sum);

    exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize)
        qk_buf[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);
}

template<typename T>
__global__ void softmax_kernel_warp_half2(half2*       qk_buf,
                                          const half2* atten_bias,
                                          const half2* atten_mask,
                                          const int    batch_size,
                                          const int    head_num,
                                          const int    seq_len)
{
    int word_id  = blockIdx.x;
    int batch_id = word_id / seq_len;
    int seq_id   = word_id % seq_len;
    // 线程束 ID，一个线程束 32 个线程，这里设置 32 个线程束，拉满了 1024 个线程在一个 block 中
    int warp_tid      = threadIdx.x;
    int head_id       = threadIdx.y;
    int half2_seq_len = seq_len / 2;  // 一个线程在长度维度处理两个 half 元素
    int qk_offset     = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

    // 共享内存数组,内存长度为 head_num*seq_len*sizeof(float)
    // 如果 head_num*seq_len 过长，则可能共享内存开辟不够，无法启动
    extern __shared__ float shmem[];
    // s_qk_buf 代表当前某个头的 seq_len 的所有值
    float* s_qk_buf = (float*)shmem + head_id * seq_len;

    float max_val = -1e20f;
    // 线程在序列长度（[B,L, NH,L] 最后一维）方向迭代，步伐为线程束的大小
    for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize) {
        half2 qk = qk_buf[qk_offset + col_id];
        if (atten_bias)
            qk = __hadd2(qk, atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]);
        // mask_val 0 代表掩码， 1 代表不动
        half2 mask_val   = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
        float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f, mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
        float tmp_x = (float)qk.x + mask_val_x, tmp_y = (float)qk.y + mask_val_y;
        s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
        max_val = fmax(max_val, fmax(tmp_x, tmp_y));
    }
    // 线程束同步，得到最后一维上的最大值
    max_val = warpReduceMax(max_val);

    float exp_sum = 0.0f;
    for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
        float qk         = __expf(s_qk_buf[col_id] - max_val);
        s_qk_buf[col_id] = qk;
        exp_sum += qk;
    }
    exp_sum = warpReduceSum(exp_sum);

    exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize)
        qk_buf[qk_offset + col_id] =
            __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum), (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
}

template<typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register(half2*       qk_buf,
                                                   const half2* atten_bias,
                                                   const half2* atten_mask,
                                                   const int    batch_size,
                                                   const int    head_num,
                                                   const int    seq_len)
{
    int word_id       = blockIdx.x;
    int batch_id      = word_id / seq_len;
    int seq_id        = word_id % seq_len;
    int warp_tid      = threadIdx.x;
    int head_id       = threadIdx.y;
    int half2_seq_len = seq_len / 2;
    int qk_offset     = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

    float s_qk_buf[count];
    if (need_padding)
        s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

    float max_val = -1e20f;
    for (int i = 0; i < count / 2; i++) {
        int col_id = warp_tid + warpSize * i;
        if (need_padding && col_id >= half2_seq_len)
            break;

        half2 qk = qk_buf[qk_offset + col_id];
        if (atten_bias)
            qk = __hadd2(qk, __ldg(&atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]));
        half2 mask_val   = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
        float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f, mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
        s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
    }

    for (int i = 0; i < count; i++)
        max_val = fmax(max_val, s_qk_buf[i]);
    max_val = warpReduceMax(max_val);

    float exp_sum = 0.0f;
    for (int i = 0; i < count; i++) {
        s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
        exp_sum += s_qk_buf[i];
    }
    exp_sum = warpReduceSum(exp_sum);

    exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int i = 0; i < count / 2; i++) {
        int col_id = warp_tid + warpSize * i;
        if (need_padding && col_id >= half2_seq_len)
            return;
        qk_buf[qk_offset + col_id] =
            __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
    }
}

#define SOFTMAX_HALF2_REG(REG_COUNT)                                                                                   \
    if (seq_len % 64 == 0)                                                                                             \
        softmax_kernel_warp_half2_register<half2, REG_COUNT, false><<<grid, block, 0, stream>>>(                       \
            (half2*)qk_buf, (half2*)atten_bias, (half2*)atten_mask, batch_size, head_num, seq_len);                    \
    else                                                                                                               \
        softmax_kernel_warp_half2_register<half2, REG_COUNT, true><<<grid, block, 0, stream>>>(                        \
            (half2*)qk_buf, (half2*)atten_bias, (half2*)atten_mask, batch_size, head_num, seq_len)

template<OperationType OpType, typename T>
void softmax_kernelLauncher(T*           qk_buf,
                            const T*     atten_bias,
                            const T*     atten_mask,
                            const int    batch_size,
                            const int    seq_len,
                            const int    head_num,
                            cudaStream_t stream)
{
    dim3 grid(batch_size * seq_len), block(32, head_num);

    const int shmem_size = head_num * seq_len * sizeof(float);
    if (shmem_size > 64 * 1024)
        printf("Not Enough Shared Memory for Softmax\n");

    if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
        // 长度为偶数，且小于等于 1024，计算类型和 fp16
        if (seq_len <= 1024) {
            switch ((seq_len + 63) / 64) {
                case 1:  // (0 ~ 64]
                    SOFTMAX_HALF2_REG(1 * 2);
                    break;
                case 2:  // (64 ~ 128]
                    SOFTMAX_HALF2_REG(2 * 2);
                    break;
                case 3:
                    SOFTMAX_HALF2_REG(3 * 2);
                    break;
                case 4:
                    SOFTMAX_HALF2_REG(4 * 2);
                    break;
                case 5:
                    SOFTMAX_HALF2_REG(5 * 2);
                    break;
                case 6:
                    SOFTMAX_HALF2_REG(6 * 2);
                    break;
                case 7:
                    SOFTMAX_HALF2_REG(7 * 2);
                    break;
                case 8:
                    SOFTMAX_HALF2_REG(8 * 2);
                    break;
                case 9:
                    SOFTMAX_HALF2_REG(9 * 2);
                    break;
                case 10:
                    SOFTMAX_HALF2_REG(10 * 2);
                    break;
                case 11:
                    SOFTMAX_HALF2_REG(11 * 2);
                    break;
                case 12:
                    SOFTMAX_HALF2_REG(12 * 2);
                    break;
                case 13:
                    SOFTMAX_HALF2_REG(13 * 2);
                    break;
                case 14:
                    SOFTMAX_HALF2_REG(14 * 2);
                    break;
                case 15:
                    SOFTMAX_HALF2_REG(15 * 2);
                    break;
                case 16:
                    SOFTMAX_HALF2_REG(16 * 2);
                    break;
            }
        }
        else {
            // 长度为偶数，但大于 1024 的情况，计算类型为 fp16，按 half2 的方式向量化操作
            if (shmem_size > 48 * 1024)
                cudaFuncSetAttribute(
                    softmax_kernel_warp_half2<half2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            softmax_kernel_warp_half2<half2><<<grid, block, shmem_size, stream>>>(
                (half2*)qk_buf, (half2*)atten_bias, (half2*)atten_mask, batch_size, head_num, seq_len);
        }
    }
    else {
        // 长度非偶数的情况
        if (shmem_size > 48 * 1024)
            cudaFuncSetAttribute(softmax_kernel_warp<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        softmax_kernel_warp<T>
            <<<grid, block, shmem_size, stream>>>(qk_buf, atten_bias, atten_mask, batch_size, head_num, seq_len);
    }
}

}  // namespace lyradiff