#include "XLUNetMidBlock2DCrossAttn.h"

namespace lyradiff {

template<typename T>
XLUNetMidBlock2DCrossAttn<T>::XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                                                        const size_t     temb_channels,
                                                        const size_t     ngroups,
                                                        const bool       use_swish,
                                                        const size_t     num_head,
                                                        const size_t     encoder_hidden_dim,
                                                        const size_t     inner_trans_num,
                                                        cudnnHandle_t    cudnn_handle,
                                                        cudaStream_t     stream,
                                                        cudaStream_t     stream_assistant,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        const bool       is_free_buffer_after_forward,
                                                        const bool       sparse,
                                                        LyraQuantType    quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    in_channels_(in_channels),
    out_channels_(in_channels),
    temb_channels_(temb_channels),
    ngroups_(ngroups),
    use_swish_(use_swish),
    num_head_(num_head),
    encoder_hidden_dim_(encoder_hidden_dim),
    inner_trans_num_(inner_trans_num),
    cudnn_handle_(cudnn_handle),
    stream_assistant_(stream_assistant)
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

    dim_per_head_ = out_channels_ / num_head_;

    resnet_ = new Resnet2DBlock<T>(in_channels_,
                                   out_channels_,
                                   ngroups,
                                   ngroups,
                                   use_swish,
                                   temb_channels_,
                                   cudnn_handle_,
                                   stream_,
                                   stream_assistant_,
                                   // stream_,
                                   cublas_wrapper_,
                                   allocator_,
                                   is_free_buffer_after_forward_);

    attn_ = new XLTransformer2dBlock<T>(out_channels_,
                                        num_head_,
                                        dim_per_head_,
                                        encoder_hidden_dim_,
                                        ngroups,
                                        inner_trans_num_,
                                        cudnn_handle_,
                                        stream_,
                                        cublas_wrapper_,
                                        allocator_,
                                        is_free_buffer_after_forward_,
                                        quant_level_);
}

template<typename T>
XLUNetMidBlock2DCrossAttn<T>::XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                                                        const size_t     temb_channels,
                                                        const size_t     ngroups,
                                                        const size_t     num_head,
                                                        const size_t     encoder_hidden_dim,
                                                        const size_t     inner_trans_num,
                                                        cudnnHandle_t    cudnn_handle,
                                                        cudaStream_t     stream,
                                                        cudaStream_t     stream_assistant,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        const bool       is_free_buffer_after_forward,
                                                        const bool       sparse,
                                                        LyraQuantType    quant_level):
    XLUNetMidBlock2DCrossAttn(in_channels,
                              temb_channels,
                              ngroups,
                              true,
                              num_head,
                              encoder_hidden_dim,
                              inner_trans_num,
                              cudnn_handle,
                              stream,
                              stream_assistant,
                              cublas_wrapper,
                              allocator,
                              is_free_buffer_after_forward,
                              sparse,
                              quant_level)
{
}
template<typename T>
XLUNetMidBlock2DCrossAttn<T>::XLUNetMidBlock2DCrossAttn(const size_t     in_channels,
                                                        const size_t     ngroups,
                                                        const size_t     num_head,
                                                        const size_t     encoder_hidden_dim,
                                                        cudnnHandle_t    cudnn_handle,
                                                        cudaStream_t     stream,
                                                        cudaStream_t     stream_assistant,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        const bool       is_free_buffer_after_forward,
                                                        const bool       sparse,
                                                        LyraQuantType    quant_level):
    XLUNetMidBlock2DCrossAttn(in_channels,
                              1280,
                              ngroups,
                              true,
                              num_head,
                              encoder_hidden_dim,
                              10,
                              cudnn_handle,
                              stream,
                              stream_assistant,
                              cublas_wrapper,
                              allocator,
                              is_free_buffer_after_forward,
                              sparse,
                              quant_level)
{
}

template<typename T>
XLUNetMidBlock2DCrossAttn<T>::XLUNetMidBlock2DCrossAttn(
    XLUNetMidBlock2DCrossAttn<T> const& xlunet_mid_block_2d_cross_attn):
    BaseLayer(xlunet_mid_block_2d_cross_attn.stream_,
              xlunet_mid_block_2d_cross_attn.cublas_wrapper_,
              xlunet_mid_block_2d_cross_attn.allocator_,
              xlunet_mid_block_2d_cross_attn.is_free_buffer_after_forward_,
              xlunet_mid_block_2d_cross_attn.cuda_device_prop_,
              xlunet_mid_block_2d_cross_attn.sparse_),
    in_channels_(xlunet_mid_block_2d_cross_attn.in_channels_),
    out_channels_(xlunet_mid_block_2d_cross_attn.out_channels_),
    temb_channels_(xlunet_mid_block_2d_cross_attn.temb_channels_),
    ngroups_(xlunet_mid_block_2d_cross_attn.ngroups_),
    use_swish_(xlunet_mid_block_2d_cross_attn.use_swish_),
    cudnn_handle_(xlunet_mid_block_2d_cross_attn.cudnn_handle_),
    stream_assistant_(xlunet_mid_block_2d_cross_attn.stream_assistant_)
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

    dim_per_head_ = xlunet_mid_block_2d_cross_attn.dim_per_head_;

    resnet_ = xlunet_mid_block_2d_cross_attn.resnet_;
    attn_   = xlunet_mid_block_2d_cross_attn.attn_;
}

template<typename T>
void XLUNetMidBlock2DCrossAttn<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "XLUNetMidBlock2DCrossAttn::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, size_t height, size_t width)` instead");
}

template<typename T>
void XLUNetMidBlock2DCrossAttn<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    // resnet_0
    size_t inter_buf_size_ = sizeof(T) * batch_size * height * width * out_channels_;

    inter_buf_ = (T*)allocator_->reMallocWithName("XLUNetMidBlock2DCrossAttn_inter_buf_", inter_buf_size_, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void XLUNetMidBlock2DCrossAttn<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void XLUNetMidBlock2DCrossAttn<T>::forward(
    TensorMap*                                output_tensors,
    const TensorMap*                          input_tensors,
    const XLUNetMidBlock2DCrossAttnWeight<T>* xlunet_mid_block_2d_cross_attn_weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, in_channels]
    //      encoder_hidden_states: [bs, seq_length, encode_hidden_dim],
    //      tem: [bs, 1280]

    // output tensors:
    //      output: [bs, height, width, out_channels]

    Tensor hidden_state_tensor         = input_tensors->at("hidden_states");
    Tensor encoder_hidden_state_tensor = input_tensors->at("encoder_hidden_states");
    Tensor temb_tensor                 = input_tensors->at("temb");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = hidden_state_tensor.shape[0];
    size_t height     = hidden_state_tensor.shape[1];
    size_t width      = hidden_state_tensor.shape[2];

    allocateBuffer(batch_size, height, width);

    resnet_->forward(output_tensors, input_tensors, xlunet_mid_block_2d_cross_attn_weights->resnet_0_weights_);

    TensorMap attn_input_tensor =
        TensorMap({{"hidden_states", output_tensor}, {"encoder_hidden_states", encoder_hidden_state_tensor}})
            .setContextThis(input_tensors);
    // attn_input_tensor.at("hidden_states").saveNpy("/home/mount/data/sdxl_mid_cross/data/cpp_res_0.npy");
    // attn_input_tensor.at("encoder_hidden_states").saveNpy("/home/mount/data/sdxl_mid_cross/data/cpp_encoder.npy");

    Tensor inter_buf_tensor =
        Tensor(MEMORY_GPU, hidden_state_tensor.type, {batch_size, height, width, out_channels_}, inter_buf_);
    TensorMap attn_output_tensor({{"output", inter_buf_tensor}});

    attn_->forward(&attn_output_tensor, &attn_input_tensor, xlunet_mid_block_2d_cross_attn_weights->attn_weights_);

    // attn_output_tensor.at("output").saveNpy("/home/mount/data/sdxl_mid_cross/data/cpp_attn_output.npy");

    TensorMap resnet_1_input_tensor({{"hidden_states", inter_buf_tensor}, {"temb", temb_tensor}});

    resnet_->forward(output_tensors, &resnet_1_input_tensor, xlunet_mid_block_2d_cross_attn_weights->resnet_1_weights_);
    // output_tensors->at("output").saveNpy("/home/mount/data/sdxl_mid_cross/data/cpp_res_1.npy");

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
XLUNetMidBlock2DCrossAttn<T>::~XLUNetMidBlock2DCrossAttn()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete resnet_;
    delete attn_;
}

template class XLUNetMidBlock2DCrossAttn<float>;
template class XLUNetMidBlock2DCrossAttn<half>;

}  // namespace lyradiff
