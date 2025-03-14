#include "src/lyradiff/layers/cross_attn_upblock_2d/CrossAttnUpBlock2d.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"
#include "src/lyradiff/kernels/interpolate/interpolate.h"

using namespace std;
namespace lyradiff {
template<typename T, bool ADD_UPSAMPLE>
CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::CrossAttnUpBlock2d(const size_t        in_channels,
                                                        const size_t        out_channels,
                                                        const size_t        prev_output_channel,
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
    prev_output_channel_(prev_output_channel),
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

    transformer2d_block_3 = new Transformer2dBlock<T>(out_channels_,
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

    resnet2d_block1 = new Resnet2DBlock<T>(out_channels_ + prev_output_channel_,
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

    resnet2d_block2 = new Resnet2DBlock<T>(out_channels + out_channels,
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

    resnet2d_block3 = new Resnet2DBlock<T>(out_channels_ + in_channels_,
                                           out_channels_,
                                           norm_num_groups_,
                                           norm_num_groups_,
                                           true,
                                           cudnn_handle_,
                                           stream_,
                                           stream_assistant_,
                                           cublas_wrapper,
                                           allocator,
                                           is_free_buffer_after_forward);
    if (ADD_UPSAMPLE) {
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
}

template<typename T, bool ADD_UPSAMPLE>
CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::CrossAttnUpBlock2d(
    CrossAttnUpBlock2d<T, ADD_UPSAMPLE> const& cross_attn_up_block2d):
    BaseLayer(cross_attn_up_block2d.stream_,
              cross_attn_up_block2d.cublas_wrapper_,
              cross_attn_up_block2d.allocator_,
              cross_attn_up_block2d.is_free_buffer_after_forward_,
              cross_attn_up_block2d.cuda_device_prop_,
              cross_attn_up_block2d.sparse_),
    in_channels_(cross_attn_up_block2d.in_channels_),
    out_channels_(cross_attn_up_block2d.out_channels_),
    prev_output_channel_(cross_attn_up_block2d.prev_output_channel_),
    temb_channels_(cross_attn_up_block2d.temb_channels_),
    head_num_(cross_attn_up_block2d.head_num_),
    norm_num_groups_(cross_attn_up_block2d.norm_num_groups_),
    cross_attn_dim_(cross_attn_up_block2d.cross_attn_dim_),
    cudnn_handle_(cross_attn_up_block2d.cudnn_handle_),
    stream_assistant_(cross_attn_up_block2d.stream_assistant_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    transformer2d_block_1 = cross_attn_up_block2d.transformer2d_block_1;
    transformer2d_block_2 = cross_attn_up_block2d.transformer2d_block_2;
    transformer2d_block_3 = cross_attn_up_block2d.transformer2d_block_3;
    resnet2d_block1       = cross_attn_up_block2d.resnet2d_block1;
    resnet2d_block2       = cross_attn_up_block2d.resnet2d_block2;
    resnet2d_block3       = cross_attn_up_block2d.resnet2d_block3;
    if (ADD_UPSAMPLE) {
        upsampler_conv = cross_attn_up_block2d.upsampler_conv;
    }
}

template<typename T, bool ADD_UPSAMPLE>
CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::~CrossAttnUpBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete transformer2d_block_1;
    delete transformer2d_block_2;
    delete transformer2d_block_3;
    delete resnet2d_block1;
    delete resnet2d_block2;
    delete resnet2d_block3;
    if (ADD_UPSAMPLE) {
        delete upsampler_conv;
    }
    freeBuffer();
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "CrossAttnUpBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;
    size_t cat1_size         = sizeof(T) * batch_size * height * width * (out_channels_ + prev_output_channel_);
    size_t cat2_size         = sizeof(T) * batch_size * height * width * (out_channels_ + out_channels_);
    size_t cat3_size         = sizeof(T) * batch_size * height * width * (out_channels_ + in_channels_);
    size_t interpolate_size  = sizeof(T) * batch_size * target_height * target_width * out_channels_;

    hidden_state_buf1_ =
        (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_hidden_state_buf1_", hidden_state_size, false);
    hidden_state_buf2_ =
        (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_hidden_state_buf2_", hidden_state_size, false);

    cat_buf1_ = (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_cat_buf1_", cat1_size, false);
    cat_buf2_ = (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_cat_buf2_", cat2_size, false);
    cat_buf3_ = (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_cat_buf2_", cat3_size, false);

    if (ADD_UPSAMPLE) {
        // interpolate_buf_ = (T*)allocator_->reMalloc(interpolate_buf_, interpolate_size, false);
        interpolate_buf_ =
            (T*)allocator_->reMallocWithName("CrossAttnUpBlock2d_interpolate_buf_", interpolate_size, false);
    }

    // is_allocate_buffer_ = true;
}

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf1_));
        allocator_->free((void**)(&hidden_state_buf2_));

        allocator_->free((void**)(&cat_buf1_));
        allocator_->free((void**)(&cat_buf2_));
        allocator_->free((void**)(&cat_buf3_));

        if (ADD_UPSAMPLE) {
            allocator_->free((void**)(&interpolate_buf_));
        }

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::forward(std::vector<lyradiff::Tensor>*                     output_tensors,
                                                  const std::vector<lyradiff::Tensor>*               input_tensors,
                                                  const CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>* weights)
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

template<typename T, bool ADD_UPSAMPLE>
void CrossAttnUpBlock2d<T, ADD_UPSAMPLE>::forward(TensorMap*                                       output_tensors,
                                                  const TensorMap*                                 input_tensors,
                                                  const CrossAttnUpBlock2dWeight<T, ADD_UPSAMPLE>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor temb                  = input_tensors->at("temb");
    Tensor round1_input          = input_tensors->at("round1_input");
    Tensor round2_input          = input_tensors->at("round2_input");
    Tensor round3_input          = input_tensors->at("round3_input");
    Tensor output                = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t target_height = output.shape[1];
    size_t target_width  = output.shape[2];

    allocateBuffer(batch_size, height, width, target_height, target_width);

    invokeCatByChannel(cat_buf1_,
                       init_hidden_states.getPtr<T>(),
                       round1_input.getPtr<T>(),
                       prev_output_channel_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    Tensor output_tensor1 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf1_);
    Tensor output_tensor2 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf2_);

    // round 1
    Tensor round1_input_tensor = Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height, width, prev_output_channel_ + out_channels_},
                                        cat_buf1_);

    TensorMap input_map  = TensorMap({{"hidden_states", round1_input_tensor}, {"temb", temb}});
    TensorMap output_map = TensorMap({{"output", output_tensor1}});


    resnet2d_block1->forward(&output_map, &input_map, weights->resnet_2d_block_weight1);

    input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                    .setContextThis(input_tensors);
    output_map = TensorMap({{"output", output_tensor2}});

    transformer2d_block_1->forward(&output_map, &input_map, weights->transformer2d_block_weight1);

    invokeCatByChannel(cat_buf2_,
                       output_tensor2.getPtr<T>(),
                       round2_input.getPtr<T>(),
                       out_channels_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    Tensor round2_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + out_channels_}, cat_buf2_);

    // round 2
    input_map  = TensorMap({{"hidden_states", round2_input_tensor}, {"temb", temb}});
    output_map = TensorMap({{"output", output_tensor1}});

    resnet2d_block2->forward(&output_map, &input_map, weights->resnet_2d_block_weight2);

    input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                    .setContextThis(input_tensors);
    output_map = TensorMap({{"output", output_tensor2}});

    transformer2d_block_2->forward(&output_map, &input_map, weights->transformer2d_block_weight2);

    invokeCatByChannel(cat_buf3_,
                       output_tensor2.getPtr<T>(),
                       round3_input.getPtr<T>(),
                       out_channels_,
                       in_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    Tensor round3_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, in_channels_ + out_channels_}, cat_buf3_);

    // round 3
    input_map  = TensorMap({{"hidden_states", round3_input_tensor}, {"temb", temb}});
    output_map = TensorMap({{"output", output_tensor1}});

    resnet2d_block3->forward(&output_map, &input_map, weights->resnet_2d_block_weight3);

    if (ADD_UPSAMPLE) {
        input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                        .setContextThis(input_tensors);
        output_map = TensorMap({{"output", output_tensor2}});
        transformer2d_block_3->forward(&output_map, &input_map, weights->transformer2d_block_weight3);

        // invokeInterpolateNearest(interpolate_buf_, output_tensor2.getPtr<T>(), batch_size, height, width,
        // out_channels_, 2, getStream());
        invokeInterpolateNearestToShape(interpolate_buf_,
                                        output_tensor2.getPtr<T>(),
                                        batch_size,
                                        height,
                                        width,
                                        out_channels_,
                                        target_height,
                                        target_width,
                                        getStream());

        // cout << "cur conv params, in channel: " << upsampler_conv->in_channels_ << " out channel: " <<
        // upsampler_conv->out_channels_ << " kernel: " << upsampler_conv->kernel_size_ << " stride: " <<
        // upsampler_conv->stride_  << endl; cout << "cur conv input params, n: " << batch_size << " h: " << height << "
        // w: " << width << " c: " <<  upsampler_conv->in_channels_ << endl; cout << endl;

        upsampler_conv->conv2dWithBias(output.getPtr<T>(),
                                       interpolate_buf_,
                                       weights->upsampler_weight,
                                       weights->upsampler_bias,
                                       batch_size,
                                       target_height,
                                       target_width);
    }
    else {
        input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                        .setContextThis(input_tensors);
        output_map = TensorMap({{"output", output}});
        transformer2d_block_3->forward(&output_map, &input_map, weights->transformer2d_block_weight3);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class CrossAttnUpBlock2d<float, true>;
template class CrossAttnUpBlock2d<float, false>;
template class CrossAttnUpBlock2d<half, true>;
template class CrossAttnUpBlock2d<half, false>;
}  // namespace lyradiff