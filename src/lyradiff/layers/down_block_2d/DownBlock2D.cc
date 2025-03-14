#include "DownBlock2D.h"

namespace lyradiff {

template<typename T>
DownBlock2D<T>::DownBlock2D(const size_t     in_channels,
                            const size_t     out_channels,
                            const size_t     temb_channels,
                            const size_t     ngroups,
                            const size_t     use_swish,
                            cudnnHandle_t    cudnn_handle,
                            cudaStream_t     stream,
                            cudaStream_t     stream_assistant,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator*      allocator,
                            const bool       is_free_buffer_after_forward,
                            const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    in_channels_(in_channels),
    out_channels_(out_channels),
    temb_channels_(temb_channels),
    ngroups_(ngroups),
    use_swish_(use_swish),
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

    resnet_0_ = new Resnet2DBlock<T>(in_channels_,
                                     out_channels_,
                                     ngroups,
                                     ngroups,
                                     use_swish,
                                     temb_channels_,
                                     cudnn_handle,
                                     stream_,
                                     stream_assistant_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);

    resnet_1_ = new Resnet2DBlock<T>(out_channels_,
                                     out_channels_,
                                     ngroups,
                                     ngroups,
                                     use_swish,
                                     temb_channels_,
                                     cudnn_handle,
                                     stream_,
                                     stream_assistant_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);
}

template<typename T>
DownBlock2D<T>::DownBlock2D(const size_t     in_channels,
                            const size_t     out_channels,
                            const size_t     temb_channels,
                            const size_t     ngroups,
                            cudnnHandle_t    cudnn_handle,
                            cudaStream_t     stream,
                            cudaStream_t     stream_assistant,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator*      allocator,
                            const bool       is_free_buffer_after_forward,
                            const bool       sparse):
    DownBlock2D(in_channels,
                out_channels,
                temb_channels,
                ngroups,
                true,
                cudnn_handle,
                stream,
                stream_assistant,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
DownBlock2D<T>::DownBlock2D(const size_t     in_channels,
                            const size_t     out_channels,
                            const size_t     ngroups,
                            cudnnHandle_t    cudnn_handle,
                            cudaStream_t     stream,
                            cudaStream_t     stream_assistant,
                            cublasMMWrapper* cublas_wrapper,
                            IAllocator*      allocator,
                            const bool       is_free_buffer_after_forward,
                            const bool       sparse):
    DownBlock2D(in_channels,
                out_channels,
                1280,
                ngroups,
                true,
                cudnn_handle,
                stream,
                stream_assistant,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
DownBlock2D<T>::DownBlock2D(DownBlock2D<T> const& down_block_2d):
    BaseLayer(down_block_2d.stream_,
              down_block_2d.cublas_wrapper_,
              down_block_2d.allocator_,
              down_block_2d.is_free_buffer_after_forward_,
              down_block_2d.cuda_device_prop_,
              down_block_2d.sparse_),
    in_channels_(down_block_2d.in_channels_),
    out_channels_(down_block_2d.out_channels_),
    temb_channels_(down_block_2d.temb_channels_),
    ngroups_(down_block_2d.ngroups_),
    use_swish_(down_block_2d.use_swish_),
    cudnn_handle_(down_block_2d.cudnn_handle_),
    stream_assistant_(down_block_2d.stream_assistant_)
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
    resnet_0_ = down_block_2d.resnet_0_;
    resnet_1_ = down_block_2d.resnet_1_;
}

template<typename T>
void DownBlock2D<T>::allocateBuffer()
{
}

template<typename T>
void DownBlock2D<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void DownBlock2D<T>::forward(TensorMap*                  output_tensors,
                             const TensorMap*            input_tensors,
                             const DownBlock2DWeight<T>* down_block_2d_weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, in_channels],
    //      tem: [bs, 1280]

    // output tensors:
    //      output_states_0: [bs, height, width, out_channels],
    //      output_states_1: [bs, height, width, out_channels],

    Tensor input_tensor = input_tensors->at("hidden_states");
    Tensor temb_tensor  = input_tensors->at("temb");

    Tensor output_tensor_0 = output_tensors->at("output_states_0");
    Tensor output_tensor_1 = output_tensors->at("output_states_1");

    size_t batch_size = input_tensor.shape[0];
    size_t height     = input_tensor.shape[1];
    size_t width      = input_tensor.shape[2];
    size_t nchannels  = input_tensor.shape[3];
    size_t temb_dim   = temb_tensor.shape[1];

    // ResNet 0:
    // input: [bs, height, width, in_channels]
    // output: [bs, height, width, out_channels]

    TensorMap resnet_0_output_tensor({{"output", output_tensor_0}});

    resnet_0_->forward(&resnet_0_output_tensor, input_tensors, down_block_2d_weights->resnet_0_weights);

    // ResNet 1:
    // input: [bs, height, width, out_channels]
    // output: [bs, height, width, out_channels]

    TensorMap resnet_1_output_tensor({{"output", output_tensor_1}});
    TensorMap resnet_1_input_tensor({{"hidden_states", output_tensor_0}, {"temb", temb_tensor}});

    resnet_1_->forward(&resnet_1_output_tensor, &resnet_1_input_tensor, down_block_2d_weights->resnet_1_weights);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
DownBlock2D<T>::~DownBlock2D()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete resnet_0_;
    delete resnet_1_;
}

template class DownBlock2D<float>;
template class DownBlock2D<half>;

}  // namespace lyradiff