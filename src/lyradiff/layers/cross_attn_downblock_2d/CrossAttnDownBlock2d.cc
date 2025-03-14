#include "src/lyradiff/layers/cross_attn_downblock_2d/CrossAttnDownBlock2d.h"

using namespace std;
namespace lyradiff {
template<typename T>
CrossAttnDownBlock2d<T>::CrossAttnDownBlock2d(const size_t        in_channels,
                                              const size_t        out_channels,
                                              const size_t        temb_channels,
                                              const size_t        head_num,
                                              const size_t        cross_attn_dim,
                                              const size_t        norm_num_groups,
                                              cudnnHandle_t       cudnn_handle,
                                              cudaStream_t        stream,
                                              cudaStream_t        stream_assistant,
                                              cublasMMWrapper*    cublas_wrapper,
                                              IAllocator*         allocator,
                                              bool                is_free_buffer_after_forward,
                                              const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false, quant_level),
    in_channels_(in_channels),
    out_channels_(out_channels),
    temb_channels_(temb_channels),
    head_num_(head_num),
    norm_num_groups_(norm_num_groups),
    cross_attn_dim_(cross_attn_dim),
    cudnn_handle_(cudnn_handle),
    stream_assistant_(stream_assistant)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    transformer2d_block_1 = new Transformer2dBlock<T>(out_channels_,
                                                      head_num_,
                                                      out_channels_ / head_num_,
                                                      cross_attn_dim_,
                                                      norm_num_groups_,
                                                      cudnn_handle_,
                                                      stream_,
                                                      cublas_wrapper,
                                                      allocator,
                                                      is_free_buffer_after_forward,
                                                      quant_level);

    transformer2d_block_2 = new Transformer2dBlock<T>(out_channels_,
                                                      head_num_,
                                                      out_channels_ / head_num_,
                                                      cross_attn_dim_,
                                                      norm_num_groups_,
                                                      cudnn_handle_,
                                                      stream_,
                                                      cublas_wrapper,
                                                      allocator,
                                                      is_free_buffer_after_forward,
                                                      quant_level);

    resnet2d_block1 = new Resnet2DBlock<T>(in_channels,
                                           out_channels,
                                           norm_num_groups_,
                                           norm_num_groups_,
                                           true,
                                           cudnn_handle_,
                                           stream_,
                                           stream_assistant_,
                                           cublas_wrapper,
                                           allocator,
                                           is_free_buffer_after_forward);

    resnet2d_block2 = new Resnet2DBlock<T>(out_channels,
                                           out_channels,
                                           norm_num_groups_,
                                           norm_num_groups_,
                                           true,
                                           cudnn_handle_,
                                           stream_,
                                           stream_assistant_,
                                           cublas_wrapper,
                                           allocator,
                                           is_free_buffer_after_forward);

    downsampler_conv = new Conv2d<T>(out_channels,
                                     out_channels,
                                     3,  // kernel size
                                     2,
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
CrossAttnDownBlock2d<T>::CrossAttnDownBlock2d(CrossAttnDownBlock2d<T> const& cross_attn_down_block2d):
    BaseLayer(cross_attn_down_block2d.stream_,
              cross_attn_down_block2d.cublas_wrapper_,
              cross_attn_down_block2d.allocator_,
              cross_attn_down_block2d.is_free_buffer_after_forward_,
              cross_attn_down_block2d.cuda_device_prop_,
              cross_attn_down_block2d.sparse_),
    in_channels_(cross_attn_down_block2d.in_channels_),
    out_channels_(cross_attn_down_block2d.out_channels_),
    temb_channels_(cross_attn_down_block2d.temb_channels_),
    head_num_(cross_attn_down_block2d.head_num_),
    norm_num_groups_(cross_attn_down_block2d.norm_num_groups_),
    cross_attn_dim_(cross_attn_down_block2d.cross_attn_dim_),
    cudnn_handle_(cross_attn_down_block2d.cudnn_handle_),
    stream_assistant_(cross_attn_down_block2d.stream_assistant_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    transformer2d_block_1 = cross_attn_down_block2d.transformer2d_block_1;
    transformer2d_block_2 = cross_attn_down_block2d.transformer2d_block_2;
    resnet2d_block1       = cross_attn_down_block2d.resnet2d_block1;
    resnet2d_block2       = cross_attn_down_block2d.resnet2d_block2;
    downsampler_conv      = cross_attn_down_block2d.downsampler_conv;
}

template<typename T>
CrossAttnDownBlock2d<T>::~CrossAttnDownBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete transformer2d_block_1;
    delete transformer2d_block_2;
    delete resnet2d_block1;
    delete resnet2d_block2;
    delete downsampler_conv;
    freeBuffer();
}

template<typename T>
void CrossAttnDownBlock2d<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "CrossAttnDownBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void CrossAttnDownBlock2d<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;

    // hidden_state_buf_   = (T*)allocator_->reMalloc(hidden_state_buf_, hidden_state_size, false);

    hidden_state_buf_ =
        (T*)allocator_->reMallocWithName("CrossAttnDownBlock2d_hidden_state_buf_", hidden_state_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void CrossAttnDownBlock2d<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf_));

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void CrossAttnDownBlock2d<T>::forward(std::vector<lyradiff::Tensor>*         output_tensors,
                                      const std::vector<lyradiff::Tensor>*   input_tensors,
                                      const CrossAttnDownBlock2dWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)},
                            {"encoder_hidden_states", input_tensors->at(1)},
                            {"temb", input_tensors->at(2)}});
    TensorMap output_tensor({{"round1_output", output_tensors->at(0)},
                             {"round2_output", output_tensors->at(1)},
                             {"downsample_output", output_tensors->at(2)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void CrossAttnDownBlock2d<T>::forward(TensorMap*                           output_tensors,
                                      const TensorMap*                     input_tensors,
                                      const CrossAttnDownBlock2dWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor temb                  = input_tensors->at("temb");
    Tensor round1_output         = output_tensors->at("round1_output");
    Tensor round2_output         = output_tensors->at("round2_output");
    Tensor downsample_output     = output_tensors->at("downsample_output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    allocateBuffer(batch_size, height, width);

    // round 1
    Tensor output_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf_);

    TensorMap input  = TensorMap({{"hidden_states", init_hidden_states}, {"temb", temb}});
    TensorMap output = TensorMap({{"output", output_tensor}});

    resnet2d_block1->forward(&output, &input, weights->resnet_2d_block_weight1);
    input = TensorMap({{"hidden_states", output_tensor}, {"encoder_hidden_states", encoder_hidden_states}})
                .setContextThis(input_tensors);
    output = TensorMap({{"output", round1_output}});
    transformer2d_block_1->forward(&output, &input, weights->transformer2d_block_weight1);

    // round 2
    input  = TensorMap({{"hidden_states", round1_output}, {"temb", temb}});
    output = TensorMap({{"output", output_tensor}});

    resnet2d_block2->forward(&output, &input, weights->resnet_2d_block_weight2);

    input = TensorMap({{"hidden_states", output_tensor}, {"encoder_hidden_states", encoder_hidden_states}})
                .setContextThis(input_tensors);
    output = TensorMap({{"output", round2_output}});

    transformer2d_block_2->forward(&output, &input, weights->transformer2d_block_weight2);

    // cout << "cur conv params, in channel: " << downsampler_conv->in_channels_ << " out channel: " <<
    // downsampler_conv->out_channels_ << " kernel: " << downsampler_conv->kernel_size_ << " stride: " <<
    // downsampler_conv->stride_  << endl; cout << "cur conv input params, n: " << batch_size << " h: " << height << "w:
    // " << width << " c: " <<  downsampler_conv->in_channels_ << endl; cout << endl;

    downsampler_conv->conv2dWithBias(downsample_output.getPtr<T>(),
                                     round2_output.getPtr<T>(),
                                     weights->downsampler_weight,
                                     weights->downsampler_bias,
                                     batch_size,
                                     height,
                                     width);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class CrossAttnDownBlock2d<float>;
template class CrossAttnDownBlock2d<half>;
}  // namespace lyradiff