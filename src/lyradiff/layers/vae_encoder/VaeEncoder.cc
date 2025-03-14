#include "VaeEncoder.h"

using namespace std;
namespace lyradiff {
template<typename T>
VaeEncoder<T>::VaeEncoder(const size_t     in_channels,
                          const size_t     out_channels,
                          const size_t     norm_num_groups,
                          cudnnHandle_t    cudnn_handle,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    conv_in = new Conv2d<T>(in_channels,
                            block_out_channels_[0],
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

    mid_block = new UNetMidBlock2D<T>(block_out_channels_[3],
                                      temb_channels_,
                                      norm_num_groups_,
                                      true,
                                      1,
                                      cudnn_handle_,
                                      stream_,
                                      cublas_wrapper,
                                      allocator,
                                      is_free_buffer_after_forward,
                                      false);

    down_encoder_block_0 = new DownEncoderBlock2D<T>(block_out_channels_[0],
                                                     block_out_channels_[1],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     true);

    down_encoder_block_1 = new DownEncoderBlock2D<T>(block_out_channels_[1],
                                                     block_out_channels_[2],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     true);

    down_encoder_block_2 = new DownEncoderBlock2D<T>(block_out_channels_[2],
                                                     block_out_channels_[3],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     true);

    down_encoder_block_3 = new DownEncoderBlock2D<T>(block_out_channels_[3],
                                                     block_out_channels_[3],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     false);

    conv_out = new Conv2d<T>(block_out_channels_[3],
                             out_channels_ * 2,
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
VaeEncoder<T>::VaeEncoder(VaeEncoder<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    in_channels_(other.in_channels_),
    out_channels_(other.out_channels_),
    norm_num_groups_(other.norm_num_groups_),
    cudnn_handle_(other.cudnn_handle_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    conv_in   = other.conv_in;
    conv_out  = other.conv_out;
    mid_block = other.mid_block;

    down_encoder_block_0 = other.down_encoder_block_0;
    down_encoder_block_1 = other.down_encoder_block_1;
    down_encoder_block_2 = other.down_encoder_block_2;
    down_encoder_block_3 = other.down_encoder_block_3;
}

template<typename T>
VaeEncoder<T>::~VaeEncoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete conv_in;
    delete conv_out;
    delete mid_block;
    delete down_encoder_block_0;
    delete down_encoder_block_1;
    delete down_encoder_block_2;
    delete down_encoder_block_3;
    freeBuffer();
}

template<typename T>
void VaeEncoder<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "VaeEncoder::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void VaeEncoder<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    prev_batch  = batch_size;
    prev_height = height;
    prev_width  = width;

    size_t down_block_share_buf_size_ = sizeof(T) * batch_size * block_out_channels_[0] * height * width;
    size_t mid_block_buf_size_        = sizeof(T) * batch_size * height / 8 * width / 8 * block_out_channels_[3];

    size_t gnorm_buf_size_   = sizeof(T) * batch_size * height / 8 * width / 8 * block_out_channels_[3];
    size_t gnorm_cache_size_ = sizeof(double) * batch_size * norm_num_groups_ * 2;

    // 因为upblock输入输出可以使用同一块显存，这里直接开辟一块最大的进行复用
    down_block_share_buf_ =
        (T*)allocator_->reMallocWithName("VaeEncoder_down_block_share_buf_", down_block_share_buf_size_, false);
    mid_block_buf_ = (T*)allocator_->reMallocWithName("VaeEncoder_mid_block_buf_", mid_block_buf_size_, false);
    gnorm_buf_     = (T*)allocator_->reMallocWithName("VaeEncoder_gnorm_buf_", gnorm_buf_size_, false);
    gnorm_cache_   = (double*)allocator_->reMallocWithName("VaeEncoder_gnorm_cache_", gnorm_cache_size_, false);

    size_t overall_size = 0;

    is_allocate_buffer_ = false;
}

template<typename T>
void VaeEncoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&down_block_share_buf_));
        allocator_->free((void**)(&mid_block_buf_));
        allocator_->free((void**)(&gnorm_buf_));
        allocator_->free((void**)(&gnorm_cache_));

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void VaeEncoder<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                            const std::vector<lyradiff::Tensor>* input_tensors,
                            const VaeEncoderWeight<T>*         weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void VaeEncoder<T>::forward(TensorMap*                 output_tensors,
                            const TensorMap*           input_tensors,
                            const VaeEncoderWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    allocateBuffer(batch_size, height, width);

    conv_in->conv2dWithBias(down_block_share_buf_,
                            init_hidden_states.getPtr<T>(),
                            weights->conv_in_weight,
                            weights->conv_in_bias,
                            batch_size,
                            height,
                            width);

    Tensor downblock_input_tensor  = Tensor(MEMORY_GPU,
                                           init_hidden_states.type,
                                            {batch_size, height, width, block_out_channels_[0]},
                                           down_block_share_buf_);
    Tensor downblock_output_tensor = Tensor(MEMORY_GPU,
                                            init_hidden_states.type,
                                            {batch_size, height / 2, width / 2, block_out_channels_[1]},
                                            down_block_share_buf_);

    TensorMap input_map  = TensorMap({{"hidden_states", downblock_input_tensor}});
    TensorMap output_map = TensorMap({{"output", downblock_output_tensor}});

    down_encoder_block_0->forward(&output_map, &input_map, weights->down_encoder_block_2d_weight_0);

    Tensor downblock_output_tensor_1 = Tensor(MEMORY_GPU,
                                              init_hidden_states.type,
                                              {batch_size, height / 4, width / 4, block_out_channels_[2]},
                                              down_block_share_buf_);

    input_map  = TensorMap({{"hidden_states", downblock_output_tensor}});
    output_map = TensorMap({{"output", downblock_output_tensor_1}});

    down_encoder_block_1->forward(&output_map, &input_map, weights->down_encoder_block_2d_weight_1);

    Tensor downblock_output_tensor_2 = Tensor(MEMORY_GPU,
                                              init_hidden_states.type,
                                              {batch_size, height / 8, width / 8, block_out_channels_[3]},
                                              down_block_share_buf_);

    input_map  = TensorMap({{"hidden_states", downblock_output_tensor_1}});
    output_map = TensorMap({{"output", downblock_output_tensor_2}});

    down_encoder_block_2->forward(&output_map, &input_map, weights->down_encoder_block_2d_weight_2);

    Tensor downblock_output_tensor_3 = Tensor(MEMORY_GPU,
                                              init_hidden_states.type,
                                              {batch_size, height / 8, width / 8, block_out_channels_[3]},
                                              down_block_share_buf_);

    input_map  = TensorMap({{"hidden_states", downblock_output_tensor_2}});
    output_map = TensorMap({{"output", downblock_output_tensor_3}});

    down_encoder_block_3->forward(&output_map, &input_map, weights->down_encoder_block_2d_weight_3);

    Tensor midblock_output_tensor = Tensor(MEMORY_GPU,
                                           init_hidden_states.type,
                                           {batch_size, height / 8, width / 8, block_out_channels_[3]},
                                           mid_block_buf_);

    input_map  = TensorMap({{"hidden_states", downblock_output_tensor_3}});
    output_map = TensorMap({{"output", midblock_output_tensor}});

    mid_block->forward(&output_map, &input_map, weights->unet_mid_block_2d_weight);

    invokeGroupNorm(gnorm_buf_,
                    mid_block_buf_,
                    weights->conv_norm_out_gamma,
                    weights->conv_norm_out_beta,
                    gnorm_cache_,
                    batch_size,
                    height / 8,
                    width / 8,
                    block_out_channels_[3],
                    norm_num_groups_,
                    true,
                    stream_);

    Tensor gnorm_buf_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height / 8, width / 8, block_out_channels_[3]}, gnorm_buf_);

    conv_out->conv2dWithBias(output.getPtr<T>(),
                             gnorm_buf_,
                             weights->conv_out_weight,
                             weights->conv_out_bias,
                             batch_size,
                             height / 8,
                             width / 8);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class VaeEncoder<float>;
template class VaeEncoder<half>;
}  // namespace lyradiff