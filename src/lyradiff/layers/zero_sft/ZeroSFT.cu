#include "ZeroSFT.h"

#include "src/lyradiff/kernels/activation_kernels.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/kernels/zero_sft/zero_sft_kernels.h"

using namespace std;
namespace lyradiff {

template<typename T>
ZeroSFT<T>::ZeroSFT(size_t           project_channels,
                    size_t           cond_channels,
                    size_t           concat_channels,
                    bool             is_mid_block,
                    cudnnHandle_t    cudnn_handle,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    const bool       is_free_buffer_after_forward,
                    const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    cudnn_handle_(cudnn_handle),
    project_channels_(project_channels),
    cond_channels_(cond_channels),
    concat_channels_(concat_channels),
    is_mid_block_(is_mid_block)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        LYRA_CHECK_WITH_INFO(false, "ZeroSFT::init() error wrong data type");
    }

    if (is_mid_block_ && concat_channels_ != 0) {
        LYRA_CHECK_WITH_INFO(false, "ZeroSFT::init() error: concat_channels_ has to be 0 when is mid block");
    }

    if (!is_mid_block_ && concat_channels_ == 0) {
        LYRA_CHECK_WITH_INFO(false, "ZeroSFT::init() error: concat_channels_ cannot be 0 when is not mid block");
    }

    mlp_conv = new Conv2d<T>(project_channels_,
                             nhidden,
                             3,
                             1,
                             1,
                             1,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             stream_,
                             cudnn_handle,
                             allocator);

    zero_add = new Conv2d<T>(nhidden,
                             cond_channels_ + concat_channels_,
                             3,
                             1,
                             1,
                             1,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             stream_,
                             cudnn_handle,
                             allocator);

    zero_mul = new Conv2d<T>(nhidden,
                             cond_channels_ + concat_channels_,
                             3,
                             1,
                             1,
                             1,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_TENSOR_NHWC,
                             stream_,
                             cudnn_handle,
                             allocator);

    zero_conv = new Conv2d<T>(cond_channels_,
                              project_channels_,
                              1,
                              1,
                              0,
                              0,
                              CUDNN_TENSOR_NHWC,
                              CUDNN_TENSOR_NHWC,
                              CUDNN_TENSOR_NHWC,
                              CUDNN_TENSOR_NHWC,
                              stream_,
                              cudnn_handle,
                              allocator);
}

template<typename T>
ZeroSFT<T>::ZeroSFT(ZeroSFT<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    cudnn_handle_(other.cudnn_handle_)
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
}

template<typename T>
void ZeroSFT<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "ZeroSFT::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void ZeroSFT<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    size_t hidden_state_raw_size = sizeof(T) * batch_size * height * width * (cond_channels_ + concat_channels_);
    // size_t hidden_state_1_size   = sizeof(T) * batch_size * height * width * project_channels_;
    // size_t hidden_state_2_size   = sizeof(T) * batch_size * height * width * (cond_channels + concat_channels);

    size_t actv_buf_size  = sizeof(T) * batch_size * height * width * (cond_channels_ + concat_channels_);
    size_t gamma_buf_size = sizeof(T) * batch_size * height * width * (cond_channels_ + concat_channels_);
    size_t beta_buf_size  = sizeof(T) * batch_size * height * width * (cond_channels_ + concat_channels_);

    size_t norm_cache_size = sizeof(double) * batch_size * norm_num_groups_ * 2;

    hidden_state_raw_buf_ =
        (T*)allocator_->reMallocWithName("ZeroSFT_hidden_state_raw_buf_", hidden_state_raw_size, false);
    // hidden_state_buf1_ = (T*)allocator_->reMallocWithName("ZeroSFT_hidden_state_buf1_", hidden_state_1_size, false);
    // hidden_state_buf2_ = (T*)allocator_->reMallocWithName("ZeroSFT_hidden_state_buf2_", hidden_state_2_size, false);

    // 一次性开一个大的buffer，以便后续复用
    actv_buf_  = (T*)allocator_->reMallocWithName("ZeroSFT_actv_buf_", actv_buf_size, false);
    gamma_buf_ = (T*)allocator_->reMallocWithName("ZeroSFT_gamma_buf_", gamma_buf_size, false);
    beta_buf_  = (T*)allocator_->reMallocWithName("ZeroSFT_beta_buf_", beta_buf_size, false);

    norm_cache_buf_ = (double*)allocator_->reMallocWithName("ZeroSFT_norm_cache_buf_", norm_cache_size, false);
}

template<typename T>
void ZeroSFT<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void ZeroSFT<T>::forward(TensorMap*              output_tensors,
                         TensorMap*              input_tensors,
                         const ZeroSFTWeight<T>* weights,
                         float                   control_scale)
{
    Tensor cur_round_input = input_tensors->at("cur_round_input");
    Tensor output          = output_tensors->at("output");

    size_t batch_size = cur_round_input.shape[0];
    size_t height     = cur_round_input.shape[1];
    size_t width      = cur_round_input.shape[2];

    if (input_tensors->isExist("control_hidden_states") && is_mid_block_ && input_tensors->isExist("hidden_states")) {
        LYRA_CHECK_WITH_INFO(false, "ZeroSFT put hidden_states for mid block");
    }

    if (is_mid_block_ && control_scale == 0.0) {
        LYRA_CHECK_WITH_INFO(false,
                             "when cur block is not mid block and control_scale is 0, this block should not be called");
    }

    // 解决无Controlnet情况
    if (!input_tensors->isExist("control_hidden_states") || control_scale == 0.0) {
        // Tensor cur_round_input = input_tensors->isExist("cur_round_input");
        Tensor hidden_states = input_tensors->at("hidden_states");
        invokeCatByChannel(output.getPtr<T>(),
                           hidden_states.getPtr<T>(),
                           cur_round_input.getPtr<T>(),
                           concat_channels_,
                           cond_channels_,
                           height,
                           width,
                           batch_size,
                           getStream());
    }
    // h = self.project_modules[adapter_idx](control[control_idx], h=_h, h_ori=h, control_scale = control_scale)

    Tensor control_hidden_states = input_tensors->at("control_hidden_states");
    allocateBuffer(batch_size, height, width);

    zero_conv->conv2dWithBiasWithResidual(gamma_buf_,
                                          control_hidden_states.getPtr<T>(),
                                          weights->zero_conv_weight,
                                          weights->zero_conv_bias,
                                          cur_round_input.getPtr<T>(),
                                          batch_size,
                                          height,
                                          width);

    // when this is not mid block's zero sft, we need
    // h + self.zero_conv(c)
    T* next_h_buf = gamma_buf_;
    if (!is_mid_block_) {
        // cout << "cur zero sft not mid block" << endl;
        Tensor hidden_states = input_tensors->at("hidden_states");
        // h = th.cat([h_ori, h], dim=1)
        invokeCatByChannel(output.getPtr<T>(),
                           hidden_states.getPtr<T>(),
                           gamma_buf_,
                           concat_channels_,
                           cond_channels_,
                           height,
                           width,
                           batch_size,
                           getStream());
        next_h_buf = output.getPtr<T>();
    }

    // actv = self.mlp_shared(c)
    mlp_conv->conv2dWithBias(actv_buf_,
                             control_hidden_states.getPtr<T>(),
                             weights->mlp_conv_weight,
                             weights->mlp_conv_bias,
                             batch_size,
                             height,
                             width);

    invokeGenericActivation<SiluActivation, T>(actv_buf_, actv_buf_, batch_size * height * width * nhidden, stream_);

    // gamma = self.zero_mul(actv)
    zero_mul->conv2dWithBias(
        gamma_buf_, actv_buf_, weights->zero_mul_weight, weights->zero_mul_bias, batch_size, height, width);

    // beta = self.zero_mul(actv)
    zero_add->conv2dWithBias(
        beta_buf_, actv_buf_, weights->zero_add_weight, weights->zero_add_bias, batch_size, height, width);
    // cout << "cur zero sft self.param_free_norm(h) * (gamma + 1) + beta" << endl;
    // h = self.param_free_norm(h) * (gamma + 1) + beta
    invokeGroupNorm<T>(actv_buf_,
                       //    output.getPtr<T>(),
                       next_h_buf,
                       weights->norm_gamma,
                       weights->norm_beta,
                       norm_cache_buf_,
                       batch_size,
                       height,
                       width,
                       concat_channels_ + cond_channels_,
                       norm_num_groups_,
                       false,
                       getStream());

    invokeMulGammaAndAddBeta(output.getPtr<T>(),
                             actv_buf_,
                             gamma_buf_,
                             beta_buf_,
                             batch_size,
                             height,
                             width,
                             concat_channels_ + cond_channels_);

    // h * control_scale + h_raw * (1 - control_scale)
    // cout << "cur zero sft h * control_scale + h_raw * (1 - control_scale)" << endl;
    if (control_scale != 1.0) {
        if (!is_mid_block_) {
            Tensor hidden_states = input_tensors->at("hidden_states");
            invokeCatByChannel(hidden_state_raw_buf_,
                               hidden_states.getPtr<T>(),
                               cur_round_input.getPtr<T>(),
                               concat_channels_,
                               cond_channels_,
                               height,
                               width,
                               batch_size,
                               getStream());

            // h * control_scale + h_raw * (1 - control_scale)
            invokeMulControlScale(output.getPtr<T>(),
                                  output.getPtr<T>(),
                                  hidden_state_raw_buf_,
                                  batch_size,
                                  height,
                                  width,
                                  concat_channels_ + cond_channels_,
                                  control_scale);
        }
        else {
            invokeMulControlScale(output.getPtr<T>(),
                                  output.getPtr<T>(),
                                  cur_round_input.getPtr<T>(),
                                  batch_size,
                                  height,
                                  width,
                                  cond_channels_,
                                  control_scale);
        }
    }
    // Tensor cur_round_input = input_tensors->at("cur_round_input");

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
ZeroSFT<T>::~ZeroSFT()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class ZeroSFT<float>;
template class ZeroSFT<half>;
}  // namespace lyradiff