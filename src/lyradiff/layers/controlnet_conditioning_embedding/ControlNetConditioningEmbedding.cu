#include "ControlNetConditioningEmbedding.h"
#include "src/lyradiff/kernels/activation_kernels.h"

using namespace std;
namespace lyradiff {

template<typename T>
ControlNetConditioningEmbedding<T>::ControlNetConditioningEmbedding(const size_t     conditioning_channels,
                                                                    const size_t     conditioning_embedding_channels,
                                                                    cudnnHandle_t    cudnn_handle,
                                                                    cudaStream_t     stream,
                                                                    cublasMMWrapper* cublas_wrapper,
                                                                    IAllocator*      allocator,
                                                                    const bool       is_free_buffer_after_forward,
                                                                    const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    conditioning_channels_(conditioning_channels),
    conditioning_embedding_channels_(conditioning_embedding_channels),
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

    int i       = 0;
    input_conv_ = new Conv2d<T>(conditioning_channels_,
                                block_out_channels_[i],
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

    for (; i < block_out_channels_.size() - 1; i++) {
        size_t channel_in  = block_out_channels_[i];
        size_t channel_out = block_out_channels_[i + 1];

        block_convs_[i * 2] = new Conv2d<T>(channel_in,
                                            channel_in,
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

        block_convs_[i * 2 + 1] = new Conv2d<T>(channel_in,
                                                channel_out,
                                                3,
                                                2,
                                                1,
                                                1,
                                                CUDNN_TENSOR_NHWC,
                                                CUDNN_TENSOR_NHWC,
                                                CUDNN_TENSOR_NHWC,
                                                CUDNN_TENSOR_NHWC,
                                                stream_,
                                                cudnn_handle,
                                                allocator);
    }

    output_conv_ = new Conv2d<T>(block_out_channels_[i],
                                 conditioning_embedding_channels_,
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
}

template<typename T>
ControlNetConditioningEmbedding<T>::ControlNetConditioningEmbedding(ControlNetConditioningEmbedding<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    conditioning_channels_(other.conditioning_channels_),
    conditioning_embedding_channels_(other.conditioning_embedding_channels_),
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
    input_conv_  = other.input_conv_;
    output_conv_ = other.output_conv_;
    block_convs_ = other.block_convs_;
}

template<typename T>
void ControlNetConditioningEmbedding<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "ControlNetConditioningEmbedding::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void ControlNetConditioningEmbedding<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    int    i                = 0;
    size_t conv_in_buf_size = sizeof(T) * batch_size * height * width * block_out_channels_[i];

    conv_in_buf_ = (T*)allocator_->reMalloc(conv_in_buf_, conv_in_buf_size, false);

    for (; i < block_out_channels_.size() - 1; i++) {
        size_t channel_in  = block_out_channels_[i];
        size_t channel_out = block_out_channels_[i + 1];

        size_t conv_buf_size = sizeof(T) * batch_size * height * width * channel_in;

        block_bufs_[i * 2] = (T*)allocator_->reMalloc(block_bufs_[i * 2], conv_buf_size, false);

        // 因为第二个blocks 的stride 是2，这里要把 height 和width 除2
        height = height / 2;
        width  = width / 2;

        size_t conv_buf2_size = sizeof(T) * batch_size * height * width * channel_out;

        block_bufs_[i * 2 + 1] = (T*)allocator_->reMalloc(block_bufs_[i * 2 + 1], conv_buf2_size, false);
    }
}

template<typename T>
void ControlNetConditioningEmbedding<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void**)(&conv_in_buf_));

    for (int i = 0; i < block_bufs_.size(); i++) {
        allocator_->free((void**)(&block_bufs_[i]));
        block_bufs_[i] = nullptr;
    }
}

template<typename T>
void ControlNetConditioningEmbedding<T>::forward(TensorMap*                                      output_tensors,
                                                 const TensorMap*                                input_tensors,
                                                 const ControlNetConditioningEmbeddingWeight<T>* weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, channels],
    //      tem: [bs, 1280]

    // output tensors:
    //      output_states_0: [bs, height, width, out_channels],
    //      output_states_1: [bs, height, width, out_channels],

    Tensor input_tensor = input_tensors->at("conditioning_img");

    Tensor output_tensor = output_tensors->at("output");

    size_t batch_size = input_tensor.shape[0];
    size_t height     = input_tensor.shape[1];
    size_t width      = input_tensor.shape[2];

    allocateBuffer(batch_size, height, width);

    // cout << "before ControlNetConditioningEmbedding input_conv_" << endl;
    input_conv_->conv2dWithBias(conv_in_buf_,
                                input_tensor.getPtr<T>(),
                                weights->conv_in_weight,
                                weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    // cout << "after ControlNetConditioningEmbedding input_conv_" << endl;

    int i = 0;
    invokeGenericActivation<SiluActivation, T>(
        conv_in_buf_, conv_in_buf_, batch_size * height * width * block_out_channels_[i], getStream());

    // cout << "after ControlNetConditioningEmbedding invokeGenericActivation" << endl;

    for (; i < block_out_channels_.size() - 1; i++) {
        if (i == 0) {
            block_convs_[0]->conv2dWithBias(block_bufs_[0],
                                            conv_in_buf_,
                                            weights->conv_block_weights[0],
                                            weights->conv_block_bias[0],
                                            batch_size,
                                            height,
                                            width);
            // cout << "after ControlNetConditioningEmbedding " << i << " conv2d" << endl;
        }
        else {
            block_convs_[i * 2]->conv2dWithBias(block_bufs_[i * 2],
                                                block_bufs_[i * 2 - 1],
                                                weights->conv_block_weights[i * 2],
                                                weights->conv_block_bias[i * 2],
                                                batch_size,
                                                height,
                                                width);

            // cout << "after ControlNetConditioningEmbedding " << i * 2 << " conv2d" << endl;
        }

        invokeGenericActivation<SiluActivation, T>(
            block_bufs_[i * 2], block_bufs_[i * 2], batch_size * height * width * block_out_channels_[i], getStream());

        block_convs_[i * 2 + 1]->conv2dWithBias(block_bufs_[i * 2 + 1],
                                                block_bufs_[i * 2],
                                                weights->conv_block_weights[i * 2 + 1],
                                                weights->conv_block_bias[i * 2 + 1],
                                                batch_size,
                                                height,
                                                width);

        // cout << "after ControlNetConditioningEmbedding " << i * 2 + 1 << " conv2d" << endl;

        height = height / 2;
        width  = width / 2;

        invokeGenericActivation<SiluActivation, T>(block_bufs_[i * 2 + 1],
                                                   block_bufs_[i * 2 + 1],
                                                   batch_size * height * width * block_out_channels_[i + 1],
                                                   getStream());
    }

    // cout << "before ControlNetConditioningEmbedding output_conv_" << endl;

    output_conv_->conv2dWithBias(output_tensor.getPtr<T>(),
                                 block_bufs_[i * 2 - 1],
                                 weights->conv_out_weight,
                                 weights->conv_out_bias,
                                 batch_size,
                                 height,
                                 width);

    // cout << "after ControlNetConditioningEmbedding output_conv_" << endl;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
ControlNetConditioningEmbedding<T>::~ControlNetConditioningEmbedding()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
    delete input_conv_;
    delete output_conv_;
    input_conv_  = nullptr;
    output_conv_ = nullptr;

    for (int i = 0; i < block_convs_.size(); i++) {
        delete block_convs_[i];
        block_convs_[i] = nullptr;
    }
}

template class ControlNetConditioningEmbedding<float>;
template class ControlNetConditioningEmbedding<half>;
}  // namespace lyradiff