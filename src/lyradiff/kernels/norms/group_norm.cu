#include "cub/cub.cuh"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"

namespace lyradiff {

template<typename T, size_t NUM_THREADS_PER_BLOCK>
__global__ void groupNormSumKernel(const T*     input,
                                   double*      caches,
                                   const size_t num_channels,
                                   const size_t num_groups,
                                   const size_t height_mul_width,
                                   const size_t hw_per_block,
                                   const size_t channel_per_block,
                                   const size_t channel_per_group,
                                   const size_t num_elems_per_batch,
                                   const size_t group_per_block)
{
    typedef cub::BlockScan<GroupSums, NUM_THREADS_PER_BLOCK> BlockScan;

    // 初始化一个共享内存变量给 BlockScan 做后续分片前缀和操作用于临时存储
    __shared__ typename BlockScan::TempStorage tempStorage;

    // 线程块中线程数量大小的共享内存数组
    __shared__ float2 shared_memory[NUM_THREADS_PER_BLOCK];

    // grid x, y, z 维度分别代表 chanel height*width  batch
    int batch_idx   = blockIdx.z;
    int channel_idx = blockIdx.x * channel_per_block + threadIdx.x * 2;

    int hw_begin_idx = blockIdx.y * hw_per_block;
    int hw_end_idx   = min(hw_begin_idx + hw_per_block, height_mul_width);

    float block_sum        = 0.0F;
    float block_sum_square = 0.0F;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    // 一个线程要处理 2 个通道 * (hw_end-hw_begin) 的元素的累加
    // 因为 GNorm 操作的分组是在 channel 维度，所以可以对 hw 维度进行分割
    // 不会导致同一个 Group 的元素处理被分配到不同的 block 而导致计算 均值方差时无法在 block 间做到 reduce
    for (int i = hw_begin_idx; i < hw_end_idx; ++i) {
        // int64_t 的索引保险一点，怕大输入的情况如 128*512*512*256 这种
        int64_t thread_offset = static_cast<int64_t>(batch_idx) * num_elems_per_batch
                                + static_cast<int64_t>(i) * num_channels + channel_idx;

        // 每个线程加载数据的时候，加载 2 个通道的数据
        T2 h2;
        h2.x = 0.0;
        h2.y = 0.0;
        if (channel_idx < num_channels) {
            h2 = *reinterpret_cast<const T2*>(input + thread_offset);
        }

        // 累加操作避免数值不稳定，将 half2 转为 float2 进行累加，Trick
        float2 ele2 = cuda_cast<float2>(h2);

        block_sum += ele2.x + ele2.y;
        block_sum_square += ele2.x * ele2.x + ele2.y * ele2.y;
    }

    // 计算当前 Block 中的当前线程在这个 block 中的第几个 group
    // 并计算出该线程在算得的这个 group 中属于第几个 channel
    int group_idx            = threadIdx.x * 2 / channel_per_group;
    int channel_idx_in_group = threadIdx.x * 2 - channel_per_group * group_idx;

    // 根据当前线程是否属于 group 中的最开始的 channel, 来分情况初始化。是开始的话，标志位 1，不是的话标志位 0
    GroupSums init_value{channel_idx_in_group == 0 ? 1 : 0, block_sum, block_sum_square};

    GroupSums out;
    // cub 操作，将一个线程块中不同线程的 init_value 排成类似数组，并做前缀和累加加起来输出到 out 里
    BlockScan(tempStorage).InclusiveScan(init_value, out, GroupSumsOp());

    if (channel_idx_in_group == channel_per_group - 2)  // the last thread in the group
    {
        // 只有这个 block 中的线程且属于一个组的最后一个的线程才执行该操作
        // 因为 InclusiveScan 会做一个前缀和累加，所以这个最后一个线程会得到前面所有的和，就是该 group 的和。
        // 那 shared_memory 在不是 group_idx 操作的位置，其值为 0
        shared_memory[group_idx] = make_float2(out.sum, out.sumSq);
    }
    __syncthreads();

    // 全局的 groupidx
    int global_group_idx = blockIdx.x * group_per_block + threadIdx.x;
    if (threadIdx.x >= group_per_block || global_group_idx >= num_groups) {
        return;
    }

    // 每个线程都会做下面指令，取出共享内存中的累计和值，如果不是前面的 group_idx 的位置，值为0
    // 可保证在原子操作 atomicAdd 中对 group 的 cache 值只在 group_idx 的位置加到有效值，其余线程加0上来不影响
    // 不同的 block 可能会写到 cache 的同一个位置，用原子操作避免碰撞
    float2 sums = shared_memory[threadIdx.x];

    atomicAdd(&caches[(2 * batch_idx + 0) * num_groups + global_group_idx], sums.x);
    atomicAdd(&caches[(2 * batch_idx + 1) * num_groups + global_group_idx], sums.y);
}

template<typename T, size_t NUM_THREADS_PER_BLOCK>
__global__ void groupNormScaleAndActivateKernel(T*           dst,
                                                const T*     input,
                                                const T*     gamma,
                                                const T*     beta,
                                                double*      caches,
                                                const size_t num_channels,
                                                const size_t num_groups,
                                                const size_t height_mul_width,
                                                const size_t hw_per_block,
                                                const size_t channel_per_block,
                                                const size_t channel_per_group,
                                                const size_t num_elems_per_batch,
                                                const float  inverse_hwc,
                                                const bool   use_swish)
{
    // grid x, y, z 维度分别代表 chanel height*width  batch
    int batch_idx        = blockIdx.z;
    int channel_idx      = blockIdx.x * channel_per_block + threadIdx.x * 2;
    int global_group_idx = channel_idx / channel_per_group;

    using T2 = typename TypeConverter<T>::Type;  // float to float2, half to half2

    float group_sum        = 0.0;
    float group_sum_square = 0.0;
    if (global_group_idx < num_groups) {
        group_sum        = caches[(2 * batch_idx + 0) * num_groups + global_group_idx];
        group_sum_square = caches[(2 * batch_idx + 1) * num_groups + global_group_idx];
    }

    T2 t_gamma, t_beta;
    if (channel_idx < num_channels) {
        t_gamma = *reinterpret_cast<const T2*>(&gamma[channel_idx]);
        t_beta  = *reinterpret_cast<const T2*>(&beta[channel_idx]);
    }

    float mean            = group_sum * inverse_hwc;
    float var             = group_sum_square * inverse_hwc - (mean * mean);
    float inverse_std_dev = var <= 0.0F ? 1.F : rsqrtf(var);

    int hw_begin_idx = blockIdx.y * hw_per_block;
    int hw_end_idx   = min(hw_begin_idx + hw_per_block, height_mul_width);

    for (int i = hw_begin_idx; i < hw_end_idx; ++i) {
        int64_t thread_offset = static_cast<int64_t>(batch_idx) * num_elems_per_batch
                                + static_cast<int64_t>(i) * num_channels + channel_idx;

        // 每个线程加载数据的时候，加载 2 个通道的数据，向量化
        T2 h2;
        h2.x = 0.0;
        h2.y = 0.0;
        if (channel_idx < num_channels) {
            h2 = *reinterpret_cast<const T2*>(input + thread_offset);
        }

        float2 ele2 = cuda_cast<float2>(h2);

        ele2.x = (ele2.x - mean) * inverse_std_dev;
        ele2.y = (ele2.y - mean) * inverse_std_dev;

        ele2.x = cuda_cast<float>(t_gamma.x) * ele2.x + cuda_cast<float>(t_beta.x);
        ele2.y = cuda_cast<float>(t_gamma.y) * ele2.y + cuda_cast<float>(t_beta.y);

        // 该判断线程束中的线程一定会一致，不会导致线程束分化
        if (use_swish) {
            ele2.x = ele2.x * sigmoid(ele2.x);  // ele2.x * sigmoid(ele2.x);
            ele2.y = ele2.y * sigmoid(ele2.y);  // ele2.x * sigmoid(ele2.x);
        }

        if (channel_idx < num_channels) {
            *reinterpret_cast<T2*>(&dst[thread_offset]) = cuda_cast<T2>(ele2);
        }
    }
}

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor)
{
    int32_t maxDivisor = -1;
    for (int32_t i = 1; i <= std::sqrt(n); i++) {
        if (n % i == 0) {
            int32_t divisor1 = n / i;
            int32_t divisor2 = i;

            if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
                maxDivisor = divisor1;
            }
            if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
                maxDivisor = divisor2;
            }
        }
    }
    return maxDivisor;
}

template<typename T>
void invokeGroupNorm(T*           dst,
                     const T*     input,
                     const T*     gamma,
                     const T*     beta,
                     double*      caches,
                     const size_t batch_size,
                     const size_t height,
                     const size_t width,
                     const size_t num_channels,
                     const size_t num_groups,
                     const bool   use_swish,
                     cudaStream_t stream)
{
    size_t  channel_per_block = 320;
    int32_t max_blocks_per_hw = 1024;

    // important: don't trust caches outside. set it to zero!!
    cudaMemsetAsync(caches, 0, sizeof(double) * num_groups * batch_size * 2, stream);

    switch (num_channels) {
        case 960:
        case 1920:
            channel_per_block = 480;
            break;
        case 256:
        case 512:
            channel_per_block = 256;
            break;
        case 128:
            channel_per_block = 128;
            break;
        default:  // c 320 640 1280 2560 ---> 1 2 4 8
            channel_per_block = 320;
    }

    size_t        height_mul_width = height * width;
    const int32_t blocks_per_hw    = findMaxDivisor(height_mul_width, max_blocks_per_hw);
    size_t        hw_per_block     = divUp(height_mul_width, blocks_per_hw);

    // printf("channel_per_block: %ld\n", channel_per_block);
    // printf("height_mul_width: %ld\n", height_mul_width);
    // printf("blocks_per_hw: %ld\n", blocks_per_hw);
    // printf("hw_per_block: %ld\n", hw_per_block);

    size_t channel_per_group   = num_channels / num_groups;
    size_t num_elems_per_batch = height * width * num_channels;
    float  inverse_hwc         = 1.0F / (height * width * channel_per_group);
    size_t group_per_block     = channel_per_block / channel_per_group;

    // printf("channel_per_group: %ld\n", channel_per_group);
    // printf("num_elems_per_batch: %ld\n", num_elems_per_batch);
    // printf("inverse_hwc: %ld\n", inverse_hwc);
    // printf("group_per_block: %ld\n", group_per_block);

    dim3 grid(num_channels / channel_per_block, divUp(height_mul_width, hw_per_block), batch_size);

    // printf("grid x y z: %d %d %d\n", grid.x, grid.y, grid.z);

    switch (channel_per_block) {
        case 128 /* constant-expression */:
            /* code */
            groupNormSumKernel<T, 64><<<grid, 64, 0, stream>>>(input,
                                                               caches,
                                                               num_channels,
                                                               num_groups,
                                                               height_mul_width,
                                                               hw_per_block,
                                                               channel_per_block,
                                                               channel_per_group,
                                                               num_elems_per_batch,
                                                               group_per_block);
            // cudaStreamSynchronize(stream);
            groupNormScaleAndActivateKernel<T, 64><<<grid, 64, 0, stream>>>(dst,
                                                                            input,
                                                                            gamma,
                                                                            beta,
                                                                            caches,
                                                                            num_channels,
                                                                            num_groups,
                                                                            height_mul_width,
                                                                            hw_per_block,
                                                                            channel_per_block,
                                                                            channel_per_group,
                                                                            num_elems_per_batch,
                                                                            inverse_hwc,
                                                                            use_swish);
            break;

        case 256:
            groupNormSumKernel<T, 128><<<grid, 128, 0, stream>>>(input,
                                                                 caches,
                                                                 num_channels,
                                                                 num_groups,
                                                                 height_mul_width,
                                                                 hw_per_block,
                                                                 channel_per_block,
                                                                 channel_per_group,
                                                                 num_elems_per_batch,
                                                                 group_per_block);
            // cudaStreamSynchronize(stream);
            groupNormScaleAndActivateKernel<T, 128><<<grid, 128, 0, stream>>>(dst,
                                                                              input,
                                                                              gamma,
                                                                              beta,
                                                                              caches,
                                                                              num_channels,
                                                                              num_groups,
                                                                              height_mul_width,
                                                                              hw_per_block,
                                                                              channel_per_block,
                                                                              channel_per_group,
                                                                              num_elems_per_batch,
                                                                              inverse_hwc,
                                                                              use_swish);
            break;

        case 320:
            groupNormSumKernel<T, 160><<<grid, 160, 0, stream>>>(input,
                                                                 caches,
                                                                 num_channels,
                                                                 num_groups,
                                                                 height_mul_width,
                                                                 hw_per_block,
                                                                 channel_per_block,
                                                                 channel_per_group,
                                                                 num_elems_per_batch,
                                                                 group_per_block);
            // cudaStreamSynchronize(stream);
            groupNormScaleAndActivateKernel<T, 160><<<grid, 160, 0, stream>>>(dst,
                                                                              input,
                                                                              gamma,
                                                                              beta,
                                                                              caches,
                                                                              num_channels,
                                                                              num_groups,
                                                                              height_mul_width,
                                                                              hw_per_block,
                                                                              channel_per_block,
                                                                              channel_per_group,
                                                                              num_elems_per_batch,
                                                                              inverse_hwc,
                                                                              use_swish);
            break;
        case 480:
            groupNormSumKernel<T, 256><<<grid, 256, 0, stream>>>(input,
                                                                 caches,
                                                                 num_channels,
                                                                 num_groups,
                                                                 height_mul_width,
                                                                 hw_per_block,
                                                                 channel_per_block,
                                                                 channel_per_group,
                                                                 num_elems_per_batch,
                                                                 group_per_block);

            // cudaStreamSynchronize(stream);
            groupNormScaleAndActivateKernel<T, 256><<<grid, 256, 0, stream>>>(dst,
                                                                              input,
                                                                              gamma,
                                                                              beta,
                                                                              caches,
                                                                              num_channels,
                                                                              num_groups,
                                                                              height_mul_width,
                                                                              hw_per_block,
                                                                              channel_per_block,
                                                                              channel_per_group,
                                                                              num_elems_per_batch,
                                                                              inverse_hwc,
                                                                              use_swish);
            break;

        default:
            throw "unsupported channel size";
            break;
    }
    // cudaStreamSynchronize(stream);
}

// 为 float 和 half 做模板特化
#define INSTANTIATE_INVOKE_GROUP_NORM(T)                                                                               \
    template void invokeGroupNorm(T*           dst,                                                                    \
                                  const T*     input,                                                                  \
                                  const T*     gamma,                                                                  \
                                  const T*     beta,                                                                   \
                                  double*      caches,                                                                 \
                                  const size_t batch_size,                                                             \
                                  const size_t height,                                                                 \
                                  const size_t width,                                                                  \
                                  const size_t num_channels,                                                           \
                                  const size_t num_groups,                                                             \
                                  const bool   use_swish,                                                              \
                                  cudaStream_t stream)

INSTANTIATE_INVOKE_GROUP_NORM(float);
INSTANTIATE_INVOKE_GROUP_NORM(half);
#undef INSTANTIATE_INVOKE_GROUP_NORM

}  // namespace lyradiff