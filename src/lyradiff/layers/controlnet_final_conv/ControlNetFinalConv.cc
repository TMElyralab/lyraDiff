#include "ControlNetFinalConv.h"

using namespace std;
namespace lyradiff {

template<typename T>
ControlNetFinalConv<T>::ControlNetFinalConv(std::vector<size_t> block_out_channels,
                                            cudnnHandle_t       cudnn_handle,
                                            cudaStream_t        stream,
                                            cublasMMWrapper*    cublas_wrapper,
                                            IAllocator*         allocator,
                                            const bool          is_free_buffer_after_forward,
                                            const bool          sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
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

    block_out_channels_ = block_out_channels;

    for (int i = 0; i < block_out_channels_.size(); i++) {
        size_t channel_in = block_out_channels_[i];
        block_convs_.push_back(new Conv2d<T>(channel_in,
                                             channel_in,
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
                                             allocator));
    }
}

template<typename T>
ControlNetFinalConv<T>::ControlNetFinalConv(ControlNetFinalConv<T> const& other):
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
    cout << "assigning block convs " << endl;
    block_convs_        = other.block_convs_;
    block_out_channels_ = other.block_out_channels_;
}

template<typename T>
void ControlNetFinalConv<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "ControlNetFinalConv::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void ControlNetFinalConv<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void ControlNetFinalConv<T>::forward(std::vector<Tensor>&                output_tensors,
                                     const std::vector<Tensor>&          input_tensors,
                                     const ControlNetFinalConvWeight<T>* weights,
                                     const std::vector<float>&           controlnet_scales)
{
    if (input_tensors.size() != output_tensors.size()) {
        throw "input_tensors size and output_tensors size not match";
    }

    if (input_tensors.size() != block_convs_.size()) {
        throw "input_tensors size and block_convs_ size not match";
    }

    for (int i = 0; i < input_tensors.size(); i++) {
        size_t batch_size = input_tensors[i].shape[0];
        size_t height     = input_tensors[i].shape[1];
        size_t width      = input_tensors[i].shape[2];

        // cout << "ControlNetFinalConv input shape: " << batch_size << " " << height << " " << width << " " <<
        // input_tensors[i].shape[3] << endl; cout << "ControlNetFinalConv output_tensors shape: " <<
        // output_tensors[i].shape[0] << " " << output_tensors[i].shape[1] << " " << output_tensors[i].shape[2] << " "
        // << output_tensors[i].shape[3] << endl;

        block_convs_[i]->conv2dWithBias(output_tensors[i].getPtr<T>(),
                                        input_tensors[i].getPtr<T>(),
                                        weights->conv_block_weights[i],
                                        weights->conv_block_bias[i],
                                        batch_size,
                                        height,
                                        width,
                                        controlnet_scales[i]);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
ControlNetFinalConv<T>::~ControlNetFinalConv()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();

    for (int i = 0; i < block_convs_.size(); i++) {
        delete block_convs_[i];
        block_convs_[i] = nullptr;
    }
}

template class ControlNetFinalConv<float>;
template class ControlNetFinalConv<half>;
}  // namespace lyradiff