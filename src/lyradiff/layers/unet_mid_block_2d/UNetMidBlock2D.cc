#include "UNetMidBlock2D.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/mid_block_2d/add_bias.h"
#include "src/lyradiff/kernels/mid_block_2d/softmax.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"

using namespace std;

namespace lyradiff {

template<typename T>
UNetMidBlock2D<T>::UNetMidBlock2D(const size_t     in_channels,
                                  const size_t     temb_channels,
                                  const size_t     ngroups,
                                  const bool       use_swish,
                                  const size_t     num_head,
                                  cudnnHandle_t    cudnn_handle,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator,
                                  const bool       is_free_buffer_after_forward,
                                  const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    in_channels_(in_channels),
    temb_channels_(temb_channels),
    ngroups_(ngroups),
    use_swish_(use_swish),
    num_head_(num_head),
    cudnn_handle_(cudnn_handle)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }

    if (num_head_ != 1) {
        throw "Unsupported num_head";
    }

    dim_per_head_ = in_channels_ / num_head_;

    qk_scale = static_cast<T>(1.0f / sqrtf(dim_per_head_ * 1.0f));

    resnet_ = new Resnet2DBlock<T>(in_channels_,
                                   in_channels_,
                                   ngroups,
                                   ngroups,
                                   use_swish,
                                   temb_channels_,
                                   cudnn_handle_,
                                   stream_,
                                   stream_,
                                   cublas_wrapper_,
                                   allocator_,
                                   is_free_buffer_after_forward_,
                                   temb_channels_ > 0);
}

template<typename T>
UNetMidBlock2D<T>::UNetMidBlock2D(UNetMidBlock2D<T> const& unet_mid_block_2d):
    BaseLayer(unet_mid_block_2d.stream_,
              unet_mid_block_2d.cublas_wrapper_,
              unet_mid_block_2d.allocator_,
              unet_mid_block_2d.is_free_buffer_after_forward_,
              unet_mid_block_2d.cuda_device_prop_,
              unet_mid_block_2d.sparse_),
    in_channels_(unet_mid_block_2d.in_channels_),
    temb_channels_(unet_mid_block_2d.temb_channels_),
    ngroups_(unet_mid_block_2d.ngroups_),
    use_swish_(unet_mid_block_2d.use_swish_),
    cudnn_handle_(unet_mid_block_2d.cudnn_handle_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }

    dim_per_head_ = unet_mid_block_2d.dim_per_head_;

    resnet_ = unet_mid_block_2d.resnet_;
}

template<typename T>
void UNetMidBlock2D<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "UNetMidBlock2D::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, size_t height, size_t width)` instead");
}

template<typename T>
void UNetMidBlock2D<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    // resnet_0
    size_t overall_size = 0;

    size_t seq_len                = height * width;
    size_t self_attn_qkv_buf_size = sizeof(T) * batch_size * seq_len * in_channels_ * 3;
    size_t self_attn_qk_buf_size  = sizeof(T) * batch_size * seq_len * seq_len;
    size_t single_qkv_buf_size_   = sizeof(T) * batch_size * seq_len * in_channels_;
    size_t norm_cache_size        = sizeof(double) * batch_size * ngroups_ * 2;

    self_attn_qkv_buf_ =
        (T*)allocator_->reMallocWithName("UNetMidBlock2D_self_attn_qkv_buf_", self_attn_qkv_buf_size, false);
    self_attn_qk_buf_ = (T*)allocator_->reMallocWithName("Resnet2DBlock_inner_buf_1", self_attn_qk_buf_size, false);
    self_attn_q_buf_ = (T*)allocator_->reMallocWithName("UNetMidBlock2D_self_attn_q_buf_", single_qkv_buf_size_, false);
    self_attn_k_buf_ = (T*)allocator_->reMallocWithName("UNetMidBlock2D_self_attn_k_buf_", single_qkv_buf_size_, false);
    self_attn_v_buf_ = (T*)allocator_->reMallocWithName("UNetMidBlock2D_self_attn_v_buf_", single_qkv_buf_size_, false);
    norm_cache_buf_  = (double*)allocator_->reMallocWithName("UNetMidBlock2D_norm_cache_buf_", norm_cache_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void UNetMidBlock2D<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)(&self_attn_qkv_buf_));
        allocator_->free((void**)(&self_attn_qk_buf_));
        allocator_->free((void**)(&self_attn_q_buf_));
        allocator_->free((void**)(&self_attn_k_buf_));
        allocator_->free((void**)(&self_attn_v_buf_));
        allocator_->free((void**)(&norm_cache_buf_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void UNetMidBlock2D<T>::forward(TensorMap*                     output_tensors,
                                const TensorMap*               input_tensors,
                                const UNetMidBlock2DWeight<T>* weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, in_channels]

    // output tensors:
    //      output: [bs, height, width, in_channels]
    Tensor hidden_state_tensor = input_tensors->at("hidden_states");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = hidden_state_tensor.shape[0];
    size_t height     = hidden_state_tensor.shape[1];
    size_t width      = hidden_state_tensor.shape[2];

    allocateBuffer(batch_size, height, width);

    Tensor inter_buf_tensor =
        Tensor(MEMORY_GPU, hidden_state_tensor.type, {batch_size, height, width, in_channels_}, self_attn_q_buf_);

    // ResNet 0:
    // input: [bs, height, width, in_channels]
    // output: [bs, height, width, in_channels]
    resnet_->forward(output_tensors, input_tensors, weights->resnet_0_weights_);

    // Attn:
    // input:
    //      hidden_states: [bs, height, width, in_channels],
    // output: [bs, height, width, in_channels]

    // 计算Self Attention
    // vae attention 前置 group norm
    invokeGroupNorm<T>(self_attn_q_buf_,
                       output_tensor.getPtr<T>(),
                       weights->gnorm_gamma,
                       weights->gnorm_beta,
                       norm_cache_buf_,
                       batch_size,
                       height,
                       width,
                       in_channels_,
                       ngroups_,
                       false,
                       getStream());

    int seq_len = height * width;
    int m       = seq_len * batch_size;
    int n       = in_channels_ * 3;
    int k       = in_channels_;

    // 计算 self attention 的 qkv，不过因为head_num 固定为 1
    // [bs, seq_len, dim_per_head] -> [bs, seq_len, 3, dim_per_head]

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          weights->attention_qkv_weight,
                          k,
                          self_attn_q_buf_,
                          k,
                          self_attn_qkv_buf_,
                          n);

    // Add bias and split hidden_states to qkv, since the head num here is 1, we don't need to do the transpose
    // so input -> outputs are [bs, seq_len, 3, dim per head] -> [bs, seq_len, dim per head] * 3
    invokeFusedAddBiasSplitQKV(self_attn_q_buf_,
                               self_attn_k_buf_,
                               self_attn_v_buf_,
                               self_attn_qkv_buf_,
                               weights->attention_qkv_bias,
                               seq_len,
                               in_channels_,
                               batch_size,
                               getStream());

    m = seq_len;
    n = seq_len;
    k = in_channels_;

    // calculate Q*K
    // q: [bs, seq_len, dim_per_head], k: [bs, seq_len, dim_per_head], qk: [bs, seq_len, seq_len]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        n,
                                        m,
                                        k,
                                        self_attn_k_buf_,
                                        k,
                                        n * k,
                                        self_attn_q_buf_,
                                        k,
                                        m * k,
                                        self_attn_qk_buf_,
                                        n,
                                        m * n,
                                        batch_size);

    Tensor self_attn_qk_tensor =
        Tensor(MEMORY_GPU, hidden_state_tensor.type, {batch_size, 1, seq_len, seq_len}, self_attn_qk_buf_);

    // self_attn_qk_tensor.saveNpy("/workspace/self_attn_qk_tensor_before_softmax.npy");

    MaskedSoftmaxParam<T, T> param;
    param.attention_score = self_attn_qk_buf_;  // (batch_size, head_num, q_length, k_length)
    param.qk              = self_attn_qk_buf_;  // (batch_size, head_num, q_length, k_length)
    param.attention_mask  = nullptr;            // (batch_size, q_length, k_length)
    param.batch_size      = batch_size;
    param.q_length        = seq_len;
    param.k_length        = seq_len;
    param.num_heads       = num_head_;
    param.qk_scale        = qk_scale;
    // param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional

    invokeMaskedSoftmax(param, stream_);

    // dispatch_scaled_softmax_forward(
    //     self_attn_qk_buf_, self_attn_qk_buf_, qk_scale, seq_len, seq_len, batch_size, num_head_, stream_);

    // self_attn_qk_tensor.saveNpy("/workspace/self_attn_qk_tensor_after_softmax.npy");

    m = seq_len;
    n = in_channels_;
    k = seq_len;
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        n,
                                        m,
                                        k,
                                        self_attn_v_buf_,
                                        n,
                                        k * n,
                                        self_attn_qk_buf_,
                                        k,
                                        m * k,
                                        self_attn_k_buf_,
                                        n,
                                        m * n,
                                        batch_size);

    m = seq_len * batch_size;
    n = in_channels_;
    k = in_channels_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n,
                                             m,
                                             k,
                                             weights->attention_to_out_weight,
                                             k,
                                             self_attn_k_buf_,
                                             k,
                                             self_attn_q_buf_,
                                             n,                               // LDC
                                             weights->attention_to_out_bias,  // bias
                                             output_tensor.getPtr<T>(),       // residual
                                             1.0f,                            // alpha
                                             1.0f);                           // beta

    Tensor attention_score(
        MEMORY_GPU, hidden_state_tensor.type, {batch_size, height, width, dim_per_head_}, self_attn_q_buf_);

    // ResNet 1:
    // input: [bs, height, width, in_channels]
    // output: [bs, height, width, in_channels]
    TensorMap resnet_1_input_tensor({{"hidden_states", attention_score}});

    resnet_->forward(output_tensors, &resnet_1_input_tensor, weights->resnet_1_weights_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
UNetMidBlock2D<T>::~UNetMidBlock2D()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete resnet_;
}

template class UNetMidBlock2D<float>;
template class UNetMidBlock2D<half>;

}  // namespace lyradiff