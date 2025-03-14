#include "DownEncoderBlock2D.h"
#include "src/lyradiff/kernels/down_encoder_block_2d/down_encoder_block_2d_kernels.h"

using namespace std;
namespace lyradiff {
template<typename T>
DownEncoderBlock2D<T>::DownEncoderBlock2D(const size_t     in_channels,
                                          const size_t     out_channels,
                                          const size_t     norm_num_groups,
                                          const size_t     temb_channels,
                                          cudnnHandle_t    cudnn_handle,
                                          cudaStream_t     stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator*      allocator,
                                          bool             is_free_buffer_after_forward,
                                          bool             add_downsample):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    temb_channels_(temb_channels),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle),
    add_downsample_(add_downsample)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    resnet2d_block_1 = new Resnet2DBlock<T>(in_channels,
                                            out_channels,
                                            norm_num_groups_,
                                            norm_num_groups_,
                                            true,
                                            temb_channels_,
                                            cudnn_handle_,
                                            stream_,
                                            stream_,
                                            cublas_wrapper,
                                            allocator,
                                            is_free_buffer_after_forward,
                                            temb_channels_ > 0);

    resnet2d_block_2 = new Resnet2DBlock<T>(out_channels_,
                                            out_channels,
                                            norm_num_groups_,
                                            norm_num_groups_,
                                            true,
                                            temb_channels_,
                                            cudnn_handle_,
                                            stream_,
                                            stream_,
                                            cublas_wrapper,
                                            allocator,
                                            is_free_buffer_after_forward,
                                            temb_channels_ > 0);

    downsampler_conv = new Conv2d<T>(out_channels_,
                                     out_channels_,
                                     3,  // kernel size
                                     2,
                                     0,
                                     0,
                                     CUDNN_TENSOR_NHWC,
                                     CUDNN_TENSOR_NHWC,
                                     CUDNN_TENSOR_NHWC,
                                     CUDNN_TENSOR_NHWC,
                                     stream,
                                     cudnn_handle,
                                     allocator);
}

template<typename T>
DownEncoderBlock2D<T>::DownEncoderBlock2D(DownEncoderBlock2D<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    in_channels_(other.in_channels_),
    out_channels_(other.out_channels_),
    temb_channels_(other.temb_channels_),
    norm_num_groups_(other.norm_num_groups_),
    cudnn_handle_(other.cudnn_handle_),
    add_downsample_(other.add_downsample_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    resnet2d_block_1 = other.resnet2d_block_1;
    resnet2d_block_2 = other.resnet2d_block_2;
    downsampler_conv = other.downsampler_conv;
}

template<typename T>
DownEncoderBlock2D<T>::~DownEncoderBlock2D()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete resnet2d_block_1;
    delete resnet2d_block_2;
    delete downsampler_conv;
    freeBuffer();
}

template<typename T>
void DownEncoderBlock2D<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "DownEncoderBlock2D::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void DownEncoderBlock2D<T>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;
    size_t pad_size          = sizeof(T) * batch_size * (height + 1) * (width + 1) * out_channels_;

    hidden_state_buf_ =
        (T*)allocator_->reMallocWithName("DownEncoderBlock2D_hidden_state_buf_", hidden_state_size, false);
    if (add_downsample_) {
        hidden_state_buf2_ =
            (T*)allocator_->reMallocWithName("DownEncoderBlock2D_hidden_state_buf2_", hidden_state_size, false);
        pad_buf_ = (T*)allocator_->reMallocWithName("DownEncoderBlock2D_pad_buf_", pad_size, false);
        cudaMemsetAsync(pad_buf_, 0, pad_size, stream_);
    }

    is_allocate_buffer_ = false;
}

template<typename T>
void DownEncoderBlock2D<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf_));
        if (add_downsample_) {
            allocator_->free((void**)(&hidden_state_buf2_));
        }

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void DownEncoderBlock2D<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                                    const std::vector<lyradiff::Tensor>* input_tensors,
                                    const DownEncoderBlock2DWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void DownEncoderBlock2D<T>::forward(TensorMap*                         output_tensors,
                                    const TensorMap*                   input_tensors,
                                    const DownEncoderBlock2DWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t target_height = output.shape[1];
    size_t target_width  = output.shape[2];

    allocateBuffer(batch_size, height, width, target_height, target_width);

    // round 1
    Tensor tensor1 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf_);
    Tensor tensor2 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf2_);

    TensorMap input_tensor_map1 = TensorMap({{"hidden_states", tensor1}});

    TensorMap output_tensor_map1 = TensorMap({{"output", tensor1}});
    TensorMap output_tensor_map2 = TensorMap({{"output", tensor2}});

    resnet2d_block_1->forward(&output_tensor_map1, input_tensors, weights->resnet_2d_block_weight1);

    // tensor1.saveNpy("/workspace/down_encoder_block_2d/resnet2d_block_1_res.npy");

    // round 2

    if (!add_downsample_) {
        resnet2d_block_2->forward(output_tensors, &input_tensor_map1, weights->resnet_2d_block_weight2);

        if (is_free_buffer_after_forward_ == true) {
            freeBuffer();
        }
        return;
    }

    resnet2d_block_2->forward(&output_tensor_map2, &input_tensor_map1, weights->resnet_2d_block_weight2);

    // tensor2.saveNpy("/workspace/down_encoder_block_2d/resnet2d_block_2_res.npy");

    invokeDownSamplePad(pad_buf_, hidden_state_buf2_, batch_size, height, width, out_channels_, stream_);

    // cout << "cur conv params, in channel: " << upsampler_conv->in_channels_ << " out channel: " <<
    // upsampler_conv->out_channels_ << " kernel: " << upsampler_conv->kernel_size_ << " stride: " <<
    // upsampler_conv->stride_  << endl; cout << "cur conv input params, n: " << batch_size << " h: " << target_height
    // << " w: " << target_width << " c: " <<  upsampler_conv->in_channels_ << endl; cout << endl;

    downsampler_conv->conv2dWithBias(output.getPtr<T>(),
                                     pad_buf_,
                                     weights->downsampler_weight,
                                     weights->downsampler_bias,
                                     batch_size,
                                     height + 1,
                                     width + 1);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class DownEncoderBlock2D<float>;
template class DownEncoderBlock2D<half>;
}  // namespace lyradiff