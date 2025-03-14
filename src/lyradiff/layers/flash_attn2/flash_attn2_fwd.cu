
#include "flash_attn2_fwd.h"

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream)
{
    FP16_SWITCH(!params.is_bf16,
                [&] { HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); }); });
}

inline std::vector<uint32_t> get_row_major_stride(const std::vector<uint32_t>& shape)
{
    std::vector<uint32_t> stride(shape.size(), 1);
    int32_t               shape_ridx = shape.size() - 1;
    uint32_t              accumulate = stride[stride.size() - 1];
    for (int32_t i = shape_ridx; i > 0; --i) {
        accumulate    = shape[i] * accumulate;
        stride[i - 1] = accumulate;
    }
    return stride;
}

template<typename T>
void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const T*                     q,
                      const T*                     k,
                      const T*                     v,
                      T*                           out,
                      const std::vector<uint32_t>& q_stride,
                      const std::vector<uint32_t>& k_stride,
                      const std::vector<uint32_t>& v_stride,
                      const std::vector<uint32_t>& out_stride,
                      void*                        cu_seqlens_q_d,
                      void*                        cu_seqlens_k_d,
                      void*                        seqused_k,
                      void*                        p_d,
                      void*                        softmax_lse_d,
                      float                        p_dropout,
                      float                        softmax_scale,
                      int                          window_size_left,
                      int                          window_size_right,
                      bool                         is_causal)
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = false;  // std::is_same<T, __nv_bfloat16>::value;

    // Set the pointers and strides.
    params.q_ptr = const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(q));
    params.k_ptr = const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(k));
    params.v_ptr = const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(v));
    // params.q_ptr = (void*)(q);
    // params.k_ptr = (void*)(k);
    // params.v_ptr = (void*)(v);

    // All stride are in elements, not bytes.
    params.q_row_stride  = q_stride[q_stride.size() - 3];  // q.stride(-3);
    params.k_row_stride  = k_stride[k_stride.size() - 3];  // k.stride(-3);
    params.v_row_stride  = v_stride[v_stride.size() - 3];  // v.stride(-3);
    params.q_head_stride = q_stride[q_stride.size() - 2];  // q.stride(-2);
    params.k_head_stride = k_stride[k_stride.size() - 2];  // k.stride(-2);
    params.v_head_stride = v_stride[v_stride.size() - 2];  // v.stride(-2);

    params.o_ptr         = reinterpret_cast<cutlass::half_t*>(out);
    // params.v_ptr = (void*)(out);

    params.o_row_stride  = out_stride[out_stride.size() - 3];  // out.stride(-3);
    params.o_head_stride = out_stride[out_stride.size() - 2];  // out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q_stride[0];    // q.stride(0);
        params.k_batch_stride = k_stride[0];    // k.stride(0);
        params.v_batch_stride = v_stride[0];    // v.stride(0);
        params.o_batch_stride = out_stride[0];  // out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
    params.seqused_k    = static_cast<int*>(seqused_k);

    params.alibi_slopes_ptr = nullptr;

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b                = b;
    params.h                = h;
    params.h_k              = h_k;
    params.h_h_k_ratio      = h / h_k;
    params.seqlen_q         = seqlen_q;
    params.seqlen_k         = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d                = d;
    params.d_rounded        = d_rounded;

    // Set the different scale values.
    params.scale_softmax      = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t     = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout               = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) {
        window_size_left = seqlen_k;
    }

    if (window_size_left >= 0 && window_size_right < 0) {
        window_size_right = seqlen_k;
    }

    params.window_size_left  = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
}

template<typename T>
void invokeFlashAttn2Fwd(T*                           out,          // batch_size x seqlen_q x num_heads x head_size
                         float*                       softmax_lse,  // batch_size x num_heads x seqlen_q  float32
                         const T*                     q,            // batch_size x seqlen_q x num_heads x head_size
                         const T*                     k,            // batch_size x seqlen_k x num_heads_k x head_size
                         const T*                     v,            // batch_size x seqlen_k x num_heads_k x head_size
                         const std::vector<uint32_t>& q_shape,
                         const std::vector<uint32_t>& k_shape,
                         const std::vector<uint32_t>& v_shape,
                         const std::vector<uint32_t>& out_shape,
                         const float                  softmax_scale,
                         const bool                   is_causal,
                         cudaStream_t                 stream)
{
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    const int batch_size   = q_shape[0];
    const int seqlen_q     = q_shape[1];
    const int num_heads    = q_shape[2];
    const int head_size_og = q_shape[3];

    const int seqlen_k    = k_shape[1];
    const int num_heads_k = k_shape[2];

    auto      round_multiple    = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size         = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded  = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded  = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing

    std::vector<uint32_t> q_stride   = get_row_major_stride(q_shape);
    std::vector<uint32_t> k_stride   = get_row_major_stride(k_shape);
    std::vector<uint32_t> v_stride   = get_row_major_stride(v_shape);
    std::vector<uint32_t> out_stride = get_row_major_stride(out_shape);

    Flash_fwd_params params;
    set_params_fprop<T>(params,
                        batch_size,
                        seqlen_q,
                        seqlen_k,
                        seqlen_q_rounded,
                        seqlen_k_rounded,
                        num_heads,
                        num_heads_k,
                        head_size,
                        head_size_rounded,
                        q,
                        k,
                        v,
                        out,
                        q_stride,
                        k_stride,
                        v_stride,
                        out_stride,
                        /*cu_seqlens_q_d=*/nullptr,
                        /*cu_seqlens_k_d=*/nullptr,
                        /*seqused_k=*/nullptr,
                        /*return_softmax*/ nullptr, /*batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded*/
                        softmax_lse,
                        /*p_dropout*/ 0.0f,
                        softmax_scale,
                        -1,
                        -1,
                        is_causal);

    run_mha_fwd(params, stream);
}

#define INSTANTIATE_INVOKE_FLASH_ATTN2_FWD(T)                                                                          \
    template void invokeFlashAttn2Fwd(T*                           out,                                                \
                                      float*                       softmax_lse,                                        \
                                      const T*                     q,                                                  \
                                      const T*                     k,                                                  \
                                      const T*                     v,                                                  \
                                      const std::vector<uint32_t>& q_shape,                                            \
                                      const std::vector<uint32_t>& k_shape,                                            \
                                      const std::vector<uint32_t>& v_shape,                                            \
                                      const std::vector<uint32_t>& out_shape,                                          \
                                      const float                  softmax_scale,                                      \
                                      const bool                   is_causal,                                          \
                                      cudaStream_t                 stream)
INSTANTIATE_INVOKE_FLASH_ATTN2_FWD(half);
INSTANTIATE_INVOKE_FLASH_ATTN2_FWD(float);  // does not supported
#undef INSTANTIATE_INVOKE_FLASH_ATTN2_FWD