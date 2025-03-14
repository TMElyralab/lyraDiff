#include "flux_transformer_kernels.h"
#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/reduce.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
__global__ void catEncoderAndHidden(T*       output,
                                    const T* encoder_hidden,
                                    const T* hidden,
                                    int      batch_size,
                                    int      hidden_seq_len,
                                    int      encoder_seq_len,
                                    int      dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_dst_col_offset = batch_idx * gridDim.x * dim + seq_idx * dim;
    size_t cur_src_col_offset;

    const T* src;
    if (seq_idx < encoder_seq_len) {
        cur_src_col_offset = batch_idx * encoder_seq_len * dim + seq_idx * dim;
        src                = encoder_hidden;
    }
    else {
        int cur_seq_idx    = seq_idx - encoder_seq_len;
        cur_src_col_offset = batch_idx * hidden_seq_len * dim + cur_seq_idx * dim;
        src                = hidden;
    }

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_dst_offset = cur_dst_col_offset + idx * 2;
        size_t cur_src_offset = cur_src_col_offset + idx * 2;

        T2 cur_data                                     = *reinterpret_cast<const T2*>(src + cur_src_offset);
        *reinterpret_cast<T2*>(&output[cur_dst_offset]) = cur_data;
    }
}

template<typename T>
__global__ void spiltEncoderAndHidden(T*       hidden_out,
                                      T*       encoder_hidden_out,
                                      const T* attn_output,
                                      int      batch_size,
                                      int      total_seq_len,
                                      int      hidden_seq_len,
                                      int      encoder_seq_len,
                                      int      dim)
{
    using T2      = typename TypeConverter<T>::Type;
    int batch_idx = blockIdx.y;
    int seq_idx   = blockIdx.x;

    size_t cur_col_offset = batch_idx * total_seq_len * dim + seq_idx * dim;
    size_t cur_dst_col_offset;

    T* dst;
    if (seq_idx < encoder_seq_len) {
        cur_dst_col_offset = batch_idx * encoder_seq_len * dim + seq_idx * dim;
        dst                = encoder_hidden_out;
    }
    else {
        int cur_seq_idx    = seq_idx - encoder_seq_len;
        cur_dst_col_offset = batch_idx * hidden_seq_len * dim + cur_seq_idx * dim;
        dst                = hidden_out;
    }

    for (uint idx = threadIdx.x; idx * 2 < dim; idx += blockDim.x) {
        size_t cur_src_offset = cur_col_offset + idx * 2;
        size_t cur_dst_offset = cur_dst_col_offset + idx * 2;

        T2 cur_data = *reinterpret_cast<const T2*>(attn_output + cur_src_offset);

        *reinterpret_cast<T2*>(&dst[cur_dst_offset]) = cur_data;
    }
}

// 专用的flux single Transformer block中的算子，支持对mlp 输入的gelu，以及把attn output和mlp concat
template<typename T>
void invokeSpiltEncoderAndHidden(T*           hidden_out,
                                 T*           encoder_hidden_out,
                                 const T*     attn_output,
                                 int          batch_size,
                                 int          hidden_seq_len,
                                 int          encoder_seq_len,
                                 int          dim,
                                 cudaStream_t stream)
{
    dim3 grid((hidden_seq_len + encoder_seq_len), batch_size);
    dim3 block(std::min(dim / 2, 1024));
    spiltEncoderAndHidden<<<grid, block, 0, stream>>>(hidden_out,
                                                      encoder_hidden_out,
                                                      attn_output,
                                                      batch_size,
                                                      hidden_seq_len + encoder_seq_len,
                                                      hidden_seq_len,
                                                      encoder_seq_len,
                                                      dim);
}

// 专用的flux single Transformer block中的算子，支持对mlp 输入的gelu，以及把attn output和mlp concat
template<typename T>
void invokeCatEncoderAndHidden(T*           output,
                               const T*     encoder_hidden,
                               const T*     hidden,
                               int          batch_size,
                               int          hidden_seq_len,
                               int          encoder_seq_len,
                               int          dim,
                               cudaStream_t stream)
{
    dim3 grid((hidden_seq_len + encoder_seq_len), batch_size);
    dim3 block(std::min(dim / 2, 1024));
    catEncoderAndHidden<<<grid, block, 0, stream>>>(
        output, encoder_hidden, hidden, batch_size, hidden_seq_len, encoder_seq_len, dim);
}

#define INSTANTIATE_INVOKE_CAT_ENCODER_AND_HIDDEN(T)                                                                   \
    template void invokeCatEncoderAndHidden(T*           output,                                                       \
                                            const T*     encoder_hidden,                                               \
                                            const T*     hidden,                                                       \
                                            int          batch_size,                                                   \
                                            int          hidden_seq_len,                                               \
                                            int          encoder_seq_len,                                              \
                                            int          dim,                                                          \
                                            cudaStream_t stream)

INSTANTIATE_INVOKE_CAT_ENCODER_AND_HIDDEN(float);
INSTANTIATE_INVOKE_CAT_ENCODER_AND_HIDDEN(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_CAT_ENCODER_AND_HIDDEN(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN

#define INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN(T)                                                                 \
    template void invokeSpiltEncoderAndHidden(T*           hidden_out,                                                 \
                                              T*           encoder_hidden_out,                                         \
                                              const T*     attn_output,                                                \
                                              int          batch_size,                                                 \
                                              int          hidden_seq_len,                                             \
                                              int          encoder_seq_len,                                            \
                                              int          dim,                                                        \
                                              cudaStream_t stream)

INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN(float);
INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN(__nv_bfloat16);
#endif

#undef INSTANTIATE_INVOKE_SPLIT_ENCODER_AND_HIDDEN

}  // namespace lyradiff