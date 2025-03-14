#include "GLVCrossAttnUpBlock2d.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"
#include "src/lyradiff/kernels/interpolate/interpolate.h"

using namespace std;
namespace lyradiff {
template<typename T>
GLVCrossAttnUpBlock2d<T>::GLVCrossAttnUpBlock2d(const size_t     in_channels,
                                                const size_t     out_channels,
                                                const size_t     prev_output_channel,
                                                const size_t     temb_channels,
                                                const size_t     head_num,
                                                const size_t     cross_attn_dim,
                                                const size_t     norm_num_groups,
                                                const size_t     inner_trans_num,
                                                cudnnHandle_t    cudnn_handle,
                                                cudaStream_t     stream,
                                                cudaStream_t     stream_assistant,
                                                cublasMMWrapper* cublas_wrapper,
                                                IAllocator*      allocator,
                                                bool             is_free_buffer_after_forward,
                                                LyraQuantType    quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false, quant_level),
    in_channels_(in_channels),
    out_channels_(out_channels),
    prev_output_channel_(prev_output_channel),
    temb_channels_(temb_channels),
    head_num_(head_num),
    norm_num_groups_(norm_num_groups),
    inner_trans_num_(inner_trans_num),
    cross_attn_dim_(cross_attn_dim),
    cudnn_handle_(cudnn_handle),
    stream_assistant_(stream_assistant)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    xltransformer2d_block = new XLTransformer2dBlock<T>(out_channels_,
                                                        head_num_,
                                                        out_channels_ / head_num_,
                                                        cross_attn_dim_,
                                                        norm_num_groups_,
                                                        inner_trans_num_,
                                                        cudnn_handle_,
                                                        stream_,
                                                        cublas_wrapper,
                                                        allocator,
                                                        is_free_buffer_after_forward);

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

    project_module1 = new ZeroSFT<T>(out_channels_,
                                     out_channels_,
                                     prev_output_channel_,
                                     false,
                                     cudnn_handle,
                                     stream_,
                                     cublas_wrapper,
                                     allocator,
                                     is_free_buffer_after_forward,
                                     false);

    project_module2 = new ZeroSFT<T>(out_channels_,
                                     out_channels_,
                                     out_channels_,
                                     false,
                                     cudnn_handle,
                                     stream_,
                                     cublas_wrapper,
                                     allocator,
                                     is_free_buffer_after_forward,
                                     false);

    project_module3 = new ZeroSFT<T>(in_channels_,
                                     in_channels_,
                                     out_channels_,
                                     false,
                                     cudnn_handle,
                                     stream_,
                                     cublas_wrapper,
                                     allocator,
                                     is_free_buffer_after_forward,
                                     false);

    cross_project_module = new ZeroCrossAttn<T>(
        out_channels_, in_channels_, stream_, cublas_wrapper, allocator, is_free_buffer_after_forward, false);

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
GLVCrossAttnUpBlock2d<T>::GLVCrossAttnUpBlock2d(GLVCrossAttnUpBlock2d<T> const& xlcross_attn_up_block2d):
    BaseLayer(xlcross_attn_up_block2d.stream_,
              xlcross_attn_up_block2d.cublas_wrapper_,
              xlcross_attn_up_block2d.allocator_,
              xlcross_attn_up_block2d.is_free_buffer_after_forward_,
              xlcross_attn_up_block2d.cuda_device_prop_,
              xlcross_attn_up_block2d.sparse_),
    in_channels_(xlcross_attn_up_block2d.in_channels_),
    out_channels_(xlcross_attn_up_block2d.out_channels_),
    prev_output_channel_(xlcross_attn_up_block2d.prev_output_channel_),
    temb_channels_(xlcross_attn_up_block2d.temb_channels_),
    head_num_(xlcross_attn_up_block2d.head_num_),
    norm_num_groups_(xlcross_attn_up_block2d.norm_num_groups_),
    cross_attn_dim_(xlcross_attn_up_block2d.cross_attn_dim_),
    inner_trans_num_(xlcross_attn_up_block2d.inner_trans_num_),
    cudnn_handle_(xlcross_attn_up_block2d.cudnn_handle_),
    stream_assistant_(xlcross_attn_up_block2d.stream_assistant_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    xltransformer2d_block = xlcross_attn_up_block2d.xltransformer2d_block;
    resnet2d_block1       = xlcross_attn_up_block2d.resnet2d_block1;
    resnet2d_block2       = xlcross_attn_up_block2d.resnet2d_block2;
    resnet2d_block3       = xlcross_attn_up_block2d.resnet2d_block3;

    upsampler_conv = xlcross_attn_up_block2d.upsampler_conv;

    project_module1 = xlcross_attn_up_block2d.project_module1;
    project_module2 = xlcross_attn_up_block2d.project_module2;
    project_module3 = xlcross_attn_up_block2d.project_module3;

    cross_project_module = xlcross_attn_up_block2d.cross_project_module;
}

template<typename T>
GLVCrossAttnUpBlock2d<T>::~GLVCrossAttnUpBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete xltransformer2d_block;
    delete resnet2d_block1;
    delete resnet2d_block2;
    delete resnet2d_block3;
    delete upsampler_conv;
    delete project_module1;
    delete project_module2;
    delete project_module3;
    delete cross_project_module;
    freeBuffer();
}

template<typename T>
void GLVCrossAttnUpBlock2d<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "GLVCrossAttnUpBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void GLVCrossAttnUpBlock2d<T>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;
    size_t cat1_size         = sizeof(T) * batch_size * height * width * (out_channels_ + prev_output_channel_);
    size_t cat2_size         = sizeof(T) * batch_size * height * width * (out_channels_ + out_channels_);
    size_t cat3_size         = sizeof(T) * batch_size * height * width * (out_channels_ + in_channels_);
    size_t interpolate_size  = sizeof(T) * batch_size * target_height * target_width * out_channels_;

    hidden_state_buf1_ =
        (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_hidden_state_buf1_", hidden_state_size, false);
    hidden_state_buf2_ =
        (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_hidden_state_buf2_", hidden_state_size, false);
    cat_buf1_ = (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_cat_buf1_", cat1_size, false);
    cat_buf2_ = (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_cat_buf2_", cat2_size, false);
    cat_buf3_ = (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_cat_buf3_", cat3_size, false);
    interpolate_buf_ =
        (T*)allocator_->reMallocWithName("GLVCrossAttnUpBlock2d_interpolate_buf_", interpolate_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void GLVCrossAttnUpBlock2d<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf1_));
        allocator_->free((void**)(&hidden_state_buf2_));

        allocator_->free((void**)(&cat_buf1_));
        allocator_->free((void**)(&cat_buf2_));
        allocator_->free((void**)(&cat_buf3_));

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
void GLVCrossAttnUpBlock2d<T>::forward(std::vector<lyradiff::Tensor>*          output_tensors,
                                       const std::vector<lyradiff::Tensor>*    input_tensors,
                                       const GLVCrossAttnUpBlock2dWeight<T>* weights,
                                       const float                           control_scale)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)},
                            {"encoder_hidden_states", input_tensors->at(1)},
                            {"temb", input_tensors->at(2)},
                            {"round1_input", input_tensors->at(3)},
                            {"round2_input", input_tensors->at(4)},
                            {"round3_input", input_tensors->at(5)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights, control_scale);
}

template<typename T>
void GLVCrossAttnUpBlock2d<T>::forward(TensorMap*                            output_tensors,
                                       const TensorMap*                      input_tensors,
                                       const GLVCrossAttnUpBlock2dWeight<T>* weights,
                                       const float                           control_scale)
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

    // 1st project module or cat if no controlnet res
    if (input_tensors->isExist("round1_control") && weights->is_maintain_project_buffer) {
        Tensor round1_control = input_tensors->at("round1_control");

        Tensor cur_output = Tensor(MEMORY_GPU,
                                   init_hidden_states.type,
                                   {batch_size, height, width, prev_output_channel_ + out_channels_},
                                   cat_buf1_);

        TensorMap cur_input_map = TensorMap({{"hidden_states", init_hidden_states},
                                             {"cur_round_input", round1_input},
                                             {"control_hidden_states", round1_control}});

        TensorMap cur_output_map = TensorMap({{"output", cur_output}});

        project_module1->forward(&cur_output_map, &cur_input_map, weights->project_module_weight1, control_scale);

        // cur_output.saveNpy("/workspace/cat_buf1_.npy");
    }
    else {
        invokeCatByChannel(cat_buf1_,
                           init_hidden_states.getPtr<T>(),
                           round1_input.getPtr<T>(),
                           prev_output_channel_,
                           out_channels_,
                           height,
                           width,
                           batch_size,
                           getStream());
    }

    Tensor output_tensor1 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf1_);
    Tensor output_tensor2 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf2_);

    // temb.saveNpy("/home/mount/data/debug/up1_temb.npy");
    // encoder_hidden_states.saveNpy("/home/mount/data/debug/up1_encoder.npy");

    // round 1
    Tensor round1_input_tensor = Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height, width, prev_output_channel_ + out_channels_},
                                        cat_buf1_);

    TensorMap input_map  = TensorMap({{"hidden_states", round1_input_tensor}, {"temb", temb}});
    TensorMap output_map = TensorMap({{"output", output_tensor1}});

    // input_map.at("hidden_states").saveNpy("/home/mount/data/debug/up1_res1_input.npy");

    resnet2d_block1->forward(&output_map, &input_map, weights->resnet_2d_block_weight1);
    // output_map.at("output").saveNpy("/home/mount/data/debug/up1_res1_out.npy");

    input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                    .setContextThis(input_tensors);
    output_map = TensorMap({{"output", output_tensor2}});

    xltransformer2d_block->forward(&output_map, &input_map, weights->xltransformer2d_block_weight1);
    // output_map.at("output").saveNpy("/home/mount/data/debug/up1_xt1_out.npy");

    // 2nd project module or cat if no controlnet res
    if (input_tensors->isExist("round2_control") && weights->is_maintain_project_buffer) {
        Tensor round2_control = input_tensors->at("round2_control");

        Tensor cur_output = Tensor(
            MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + out_channels_}, cat_buf2_);

        TensorMap cur_input_map = TensorMap({{"hidden_states", output_tensor2},
                                             {"cur_round_input", round2_input},
                                             {"control_hidden_states", round2_control}});

        TensorMap cur_output_map = TensorMap({{"output", cur_output}});

        project_module2->forward(&cur_output_map, &cur_input_map, weights->project_module_weight2, control_scale);
        // cur_output.saveNpy("/workspace/cat_buf2_.npy");
    }
    else {
        invokeCatByChannel(cat_buf2_,
                           output_tensor2.getPtr<T>(),
                           round2_input.getPtr<T>(),
                           out_channels_,
                           out_channels_,
                           height,
                           width,
                           batch_size,
                           getStream());
    }

    Tensor round2_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + out_channels_}, cat_buf2_);

    // round 2
    input_map  = TensorMap({{"hidden_states", round2_input_tensor}, {"temb", temb}});
    output_map = TensorMap({{"output", output_tensor1}});
    // input_map.at("hidden_states").saveNpy("/home/mount/data/debug/up1_res2_input.npy");

    resnet2d_block2->forward(&output_map, &input_map, weights->resnet_2d_block_weight2);
    // output_map.at("output").saveNpy("/home/mount/data/debug/up1_res2_out.npy");

    input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                    .setContextThis(input_tensors);

    output_map = TensorMap({{"output", output_tensor2}});

    xltransformer2d_block->forward(&output_map, &input_map, weights->xltransformer2d_block_weight2);
    // output_map.at("output").saveNpy("/home/mount/data/debug/up1_xt2_out.npy");

    // 3rd project module or cat if no controlnet res
    if (input_tensors->isExist("round3_control") && weights->is_maintain_project_buffer) {
        Tensor round3_control = input_tensors->at("round3_control");

        Tensor cur_output = Tensor(
            MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + in_channels_}, cat_buf3_);

        TensorMap cur_input_map = TensorMap({{"hidden_states", output_tensor2},
                                             {"cur_round_input", round3_input},
                                             {"control_hidden_states", round3_control}});

        TensorMap cur_output_map = TensorMap({{"output", cur_output}});

        project_module3->forward(&cur_output_map, &cur_input_map, weights->project_module_weight3, control_scale);
        // cur_output.saveNpy("/workspace/cat_buf3_.npy");
    }
    else {
        invokeCatByChannel(cat_buf3_,
                           output_tensor2.getPtr<T>(),
                           round3_input.getPtr<T>(),
                           out_channels_,
                           in_channels_,
                           height,
                           width,
                           batch_size,
                           getStream());
    }

    Tensor round3_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, in_channels_ + out_channels_}, cat_buf3_);

    // round 3
    input_map  = TensorMap({{"hidden_states", round3_input_tensor}, {"temb", temb}});
    output_map = TensorMap({{"output", output_tensor1}});

    // input_map.at("hidden_states").saveNpy("/home/mount/data/debug/up1_res3_input.npy");

    resnet2d_block3->forward(&output_map, &input_map, weights->resnet_2d_block_weight3);
    // output_map.at("output").saveNpy("/home/mount/data/debug/up1_res3_out.npy");

    input_map = TensorMap({{"hidden_states", output_tensor1}, {"encoder_hidden_states", encoder_hidden_states}})
                    .setContextThis(input_tensors);
    output_map = TensorMap({{"output", output_tensor2}});
    xltransformer2d_block->forward(&output_map, &input_map, weights->xltransformer2d_block_weight3);

    // final project module for cross attention
    if (input_tensors->isExist("round3_control") && weights->is_maintain_project_buffer) {
        Tensor round3_control = input_tensors->at("round3_control");

        Tensor cur_output =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, cat_buf3_);

        TensorMap input_tensor({{"hidden_states", output_tensor2}, {"context", round3_control}});
        TensorMap output_tensor({{"output", cur_output}});

        cross_project_module->forward(
            &output_tensor, &input_tensor, weights->cross_project_module_weight, control_scale);

        // cur_output.saveNpy("/workspace/cat_buf4_.npy");

        invokeInterpolateNearestToShape(interpolate_buf_,
                                        cur_output.getPtr<T>(),
                                        batch_size,
                                        height,
                                        width,
                                        out_channels_,
                                        target_height,
                                        target_width,
                                        getStream());
    }
    else {
        invokeInterpolateNearestToShape(interpolate_buf_,
                                        output_tensor2.getPtr<T>(),
                                        batch_size,
                                        height,
                                        width,
                                        out_channels_,
                                        target_height,
                                        target_width,
                                        getStream());
    }

    Tensor interpolate_t = Tensor(MEMORY_GPU,
                                  init_hidden_states.type,
                                  {batch_size, target_height, target_width, out_channels_},
                                  interpolate_buf_);

    upsampler_conv->conv2dWithBias(output.getPtr<T>(),
                                   interpolate_buf_,
                                   weights->upsampler_weight,
                                   weights->upsampler_bias,
                                   batch_size,
                                   target_height,
                                   target_width);

    // output.saveNpy("/home/mount/data/debug/up1_out.npy");

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class GLVCrossAttnUpBlock2d<float>;
template class GLVCrossAttnUpBlock2d<half>;
}  // namespace lyradiff