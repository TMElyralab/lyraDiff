#include "src/lyradiff/layers/upblock2d/UpBlock2d.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"
#include "src/lyradiff/kernels/interpolate/interpolate.h"

using namespace std;
namespace lyradiff {
template<typename T>
UpBlock2d<T>::UpBlock2d(const size_t     in_channels,
                        const size_t     out_channels,
                        const size_t     prev_output_channel,
                        const size_t     norm_num_groups,
                        cudnnHandle_t    cudnn_handle,
                        cudaStream_t     stream,
                        cudaStream_t     stream_assistant,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator*      allocator,
                        bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    prev_output_channel_(prev_output_channel),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle),
    stream_assistant_(stream_assistant)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    if (in_channels_ != out_channels_ || in_channels_ != prev_output_channel_
        || out_channels_ != prev_output_channel_) {
        throw "Upblock's in_channels_, out_channels_ and prev_output_channel_ should be the same";
    }

    resnet2d_block = new Resnet2DBlock<T>(out_channels_ + out_channels_,
                                          out_channels,
                                          norm_num_groups_,
                                          norm_num_groups_,
                                          true,
                                          1280,
                                          cudnn_handle_,
                                          stream_,
                                          stream_assistant_,
                                          cublas_wrapper,
                                          allocator,
                                          is_free_buffer_after_forward);

    upsampler_conv = new Conv2d<T>(out_channels_,
                                   out_channels_,
                                   3,  // kernel size
                                   1,
                                   1,
                                   1,
                                   CUDNN_TENSOR_NHWC,
                                   CUDNN_TENSOR_NHWC,
                                   CUDNN_TENSOR_NHWC,
                                   CUDNN_TENSOR_NHWC,
                                   stream,
                                   cudnn_handle,
                                   allocator);
}

template<typename T>
UpBlock2d<T>::UpBlock2d(UpBlock2d<T> const& up_block2d):
    BaseLayer(up_block2d.stream_,
              up_block2d.cublas_wrapper_,
              up_block2d.allocator_,
              up_block2d.is_free_buffer_after_forward_,
              up_block2d.cuda_device_prop_,
              up_block2d.sparse_),
    in_channels_(up_block2d.in_channels_),
    out_channels_(up_block2d.out_channels_),
    prev_output_channel_(up_block2d.prev_output_channel_),
    norm_num_groups_(up_block2d.norm_num_groups_),
    cudnn_handle_(up_block2d.cudnn_handle_),
    stream_assistant_(up_block2d.stream_assistant_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    resnet2d_block = up_block2d.resnet2d_block;
    upsampler_conv = up_block2d.upsampler_conv;
}

template<typename T>
UpBlock2d<T>::~UpBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete resnet2d_block;
    delete upsampler_conv;
    freeBuffer();
}

template<typename T>
void UpBlock2d<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "UpBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void UpBlock2d<T>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;
    size_t cat_size          = sizeof(T) * batch_size * height * width * (out_channels_ + out_channels_);
    size_t interpolate_size  = sizeof(T) * batch_size * target_height * target_width * out_channels_;

    hidden_state_buf_ = (T*)allocator_->reMallocWithName("UpBlock2d_hidden_state_buf_", hidden_state_size, false);
    cat_buf_          = (T*)allocator_->reMallocWithName("UpBlock2d_cat_buf_", cat_size, false);
    interpolate_buf_  = (T*)allocator_->reMallocWithName("UpBlock2d_interpolate_buf_", interpolate_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void UpBlock2d<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf_));
        allocator_->free((void**)(&cat_buf_));
        allocator_->free((void**)(&interpolate_buf_));

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void UpBlock2d<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                           const std::vector<lyradiff::Tensor>* input_tensors,
                           const UpBlock2dWeight<T>*          weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)},
                            {"encoder_hidden_states", input_tensors->at(1)},
                            {"temb", input_tensors->at(2)},
                            {"round1_input", input_tensors->at(3)},
                            {"round2_input", input_tensors->at(4)},
                            {"round3_input", input_tensors->at(5)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void UpBlock2d<T>::forward(TensorMap* output_tensors, const TensorMap* input_tensors, const UpBlock2dWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor temb               = input_tensors->at("temb");
    Tensor round1_input       = input_tensors->at("round1_input");
    Tensor round2_input       = input_tensors->at("round2_input");
    Tensor round3_input       = input_tensors->at("round3_input");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t target_height = output.shape[1];
    size_t target_width  = output.shape[2];

    allocateBuffer(batch_size, height, width, target_height, target_width);

    invokeCatByChannel(cat_buf_,
                       init_hidden_states.getPtr<T>(),
                       round1_input.getPtr<T>(),
                       prev_output_channel_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 1
    Tensor input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + out_channels_}, cat_buf_);
    Tensor output_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf_);

    TensorMap input_map  = TensorMap({{"hidden_states", input_tensor}, {"temb", temb}});
    TensorMap output_map = TensorMap({{"output", output_tensor}});

    resnet2d_block->forward(&output_map, &input_map, weights->resnet_2d_block_weight1);
    invokeCatByChannel(cat_buf_,
                       output_tensor.getPtr<T>(),
                       round2_input.getPtr<T>(),
                       out_channels_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 2
    resnet2d_block->forward(&output_map, &input_map, weights->resnet_2d_block_weight2);
    invokeCatByChannel(cat_buf_,
                       output_tensor.getPtr<T>(),
                       round3_input.getPtr<T>(),
                       out_channels_,
                       in_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 3
    resnet2d_block->forward(&output_map, &input_map, weights->resnet_2d_block_weight3);

    // invokeInterpolateNearest(interpolate_buf_, output_tensor.getPtr<T>(), batch_size, height, width, out_channels_,
    // 2, getStream());
    invokeInterpolateNearestToShape(interpolate_buf_,
                                    output_tensor.getPtr<T>(),
                                    batch_size,
                                    height,
                                    width,
                                    out_channels_,
                                    target_height,
                                    target_width,
                                    getStream());

    // cout << "cur conv params, in channel: " << upsampler_conv->in_channels_ << " out channel: " <<
    // upsampler_conv->out_channels_ << " kernel: " << upsampler_conv->kernel_size_ << " stride: " <<
    // upsampler_conv->stride_  << endl; cout << "cur conv input params, n: " << batch_size << " h: " << target_height
    // << " w: " << target_width << " c: " <<  upsampler_conv->in_channels_ << endl; cout << endl;

    upsampler_conv->conv2dWithBias(output.getPtr<T>(),
                                   interpolate_buf_,
                                   weights->upsampler_weight,
                                   weights->upsampler_bias,
                                   batch_size,
                                   target_height,
                                   target_width);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class UpBlock2d<float>;
template class UpBlock2d<half>;
}  // namespace lyradiff