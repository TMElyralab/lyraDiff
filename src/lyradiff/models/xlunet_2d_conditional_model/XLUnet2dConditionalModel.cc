#include "XLUnet2dConditionalModel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/utils/test_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLUnet2dConditionalModel<T>::XLUnet2dConditionalModel(cudnnHandle_t       cudnn_handle,
                                                      cudaStream_t        stream,
                                                      cublasMMWrapper*    cublas_wrapper,
                                                      IAllocator*         allocator,
                                                      const bool          is_free_buffer_after_forward,
                                                      const bool          sparse,
                                                      const bool          use_runtime_augemb,
                                                      const size_t        input_channels,
                                                      const size_t        output_channels,
                                                      const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    cudnn_handle_(cudnn_handle),
    use_runtime_augemb_(use_runtime_augemb),
    input_channels_(input_channels),
    output_channels_(output_channels)
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

    // if (max_controlnet_num > 3) {
    //     throw "max_controlnet_num too big";
    // }
    if (use_runtime_augemb_) {
        texttime_embedding = new TextTimeEmbeddingBlock<T>(block_out_channels_[0],
                                                           add_emb_dim_,
                                                           temb_channels_,
                                                           temb_channels_,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           false);
    }
    else {

        time_proj = new TimeProjection<T>(
            block_out_channels_[0], true, 0, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false);

        timestep_embedding = new TimestepEmbeddingBlock<T>(block_out_channels_[0],
                                                           temb_channels_,
                                                           temb_channels_,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           false);
    }

    down_block_2d = new XLDownBlock2D<T>(block_out_channels_[0],
                                         block_out_channels_[0],
                                         temb_channels_,
                                         norm_num_groups_,
                                         false,
                                         true,
                                         cudnn_handle_,
                                         stream,
                                         stream,
                                         cublas_wrapper,
                                         allocator,
                                         is_free_buffer_after_forward,
                                         false);

    cross_attn_down_block_2d_1 = new XLCrossAttnDownBlock2d<T>(block_out_channels_[0],
                                                               block_out_channels_[1],
                                                               temb_channels_,
                                                               head_nums_[0],
                                                               cross_attn_dim_,
                                                               norm_num_groups_,
                                                               inner_trans_nums_[0],
                                                               true,
                                                               cudnn_handle_,
                                                               stream,
                                                               stream,
                                                               cublas_wrapper,
                                                               allocator,
                                                               is_free_buffer_after_forward,
                                                               quant_level);

    cross_attn_down_block_2d_2 = new XLCrossAttnDownBlock2d<T>(block_out_channels_[1],
                                                               block_out_channels_[2],
                                                               temb_channels_,
                                                               head_nums_[1],
                                                               cross_attn_dim_,
                                                               norm_num_groups_,
                                                               inner_trans_nums_[1],
                                                               false,
                                                               cudnn_handle_,
                                                               stream,
                                                               stream,
                                                               cublas_wrapper,
                                                               allocator,
                                                               is_free_buffer_after_forward,
                                                               quant_level);

    mid_block_2d = new XLUNetMidBlock2DCrossAttn<T>(block_out_channels_[2],
                                                    temb_channels_,
                                                    norm_num_groups_,
                                                    head_nums_[1],
                                                    cross_attn_dim_,
                                                    inner_trans_nums_[1],
                                                    cudnn_handle_,
                                                    stream,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator,
                                                    is_free_buffer_after_forward,
                                                    false,
                                                    quant_level);

    cross_attn_up_block_2d_1 = new XLCrossAttnUpBlock2d<T>(block_out_channels_[1],
                                                           block_out_channels_[2],
                                                           block_out_channels_[2],
                                                           temb_channels_,
                                                           head_nums_[1],
                                                           cross_attn_dim_,
                                                           norm_num_groups_,
                                                           inner_trans_nums_[1],
                                                           cudnn_handle_,
                                                           stream,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           quant_level);

    cross_attn_up_block_2d_2 = new XLCrossAttnUpBlock2d<T>(block_out_channels_[0],
                                                           block_out_channels_[1],
                                                           block_out_channels_[2],
                                                           temb_channels_,
                                                           head_nums_[0],
                                                           cross_attn_dim_,
                                                           norm_num_groups_,
                                                           inner_trans_nums_[0],
                                                           cudnn_handle_,
                                                           stream,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           quant_level);

    up_block_2d = new XLUpBlock2d<T>(block_out_channels_[0],
                                     block_out_channels_[0],
                                     block_out_channels_[1],
                                     norm_num_groups_,
                                     cudnn_handle_,
                                     stream,
                                     stream,
                                     cublas_wrapper,
                                     allocator,
                                     is_free_buffer_after_forward);

    input_conv_ = new Conv2d<T>(input_channels_,
                                block_out_channels_[0],
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

    output_conv_ = new Conv2d<T>(block_out_channels_[0],
                                 output_channels_,
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

    controlnet = new XLControlNetModel<T>(
        false, cudnn_handle, stream_, cublas_wrapper, allocator, is_free_buffer_after_forward, false);

    // controlnet->texttime_embedding         = texttime_embedding;
    // controlnet->cross_attn_down_block_2d_1 = cross_attn_down_block_2d_1;
    // controlnet->cross_attn_down_block_2d_2 = cross_attn_down_block_2d_2;
    // controlnet->down_block_2d              = down_block_2d;
    // controlnet->mid_block_2d               = mid_block_2d;
    // controlnet->input_conv_                = input_conv_;
}

template<typename T>
XLUnet2dConditionalModel<T>::XLUnet2dConditionalModel(XLUnet2dConditionalModel<T> const& unet):
    BaseLayer(unet.stream_,
              unet.cublas_wrapper_,
              unet.allocator_,
              unet.is_free_buffer_after_forward_,
              unet.cuda_device_prop_,
              unet.sparse_),
    cudnn_handle_(unet.cudnn_handle_),
    use_runtime_augemb_(unet.use_runtime_augemb_),
    input_channels_(unet.input_channels_),
    output_channels_(unet.output_channels_)
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
    texttime_embedding         = unet.texttime_embedding;
    down_block_2d              = unet.down_block_2d;
    cross_attn_down_block_2d_1 = unet.cross_attn_down_block_2d_1;
    cross_attn_down_block_2d_2 = unet.cross_attn_down_block_2d_2;
    mid_block_2d               = unet.mid_block_2d;
    cross_attn_up_block_2d_1   = unet.cross_attn_up_block_2d_1;
    cross_attn_up_block_2d_2   = unet.cross_attn_up_block_2d_2;
    up_block_2d                = unet.up_block_2d;
    input_conv_                = unet.input_conv_;
    output_conv_               = unet.output_conv_;
    controlnet                 = unet.controlnet;
}

template<typename T>
void XLUnet2dConditionalModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "XLUnet2dConditionalModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void XLUnet2dConditionalModel<T>::allocateBuffer(size_t batch_size,
                                                 size_t height,
                                                 size_t width,
                                                 size_t controlnet_input_count)
{
    cur_batch = batch_size;
    // cur_controlnet_count = controlnet_input_count;

    // size_t input_buf_size     = sizeof(T) * batch_size * height * width * input_channels_;
    size_t conv_buf_size      = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t temb_buf_size      = sizeof(T) * batch_size * temb_channels_;
    size_t norm_cache_size    = sizeof(double) * batch_size * norm_num_groups_ * 2;
    size_t time_proj_buf_size = sizeof(float) * batch_size * block_out_channels_[0];

    conv_buf_ = (T*)allocator_->reMalloc(conv_buf_, conv_buf_size, false);
    // input_buf_ = (T*)allocator_->reMalloc(input_buf_, input_buf_size, false);

    if (!use_runtime_augemb_) {
        time_proj_buf_ = (T*)allocator_->reMalloc(time_proj_buf_, time_proj_buf_size, false);
    }

    temb_buf_       = (T*)allocator_->reMalloc(temb_buf_, temb_buf_size, false);
    norm_cache_buf_ = (double*)allocator_->reMalloc(norm_cache_buf_, norm_cache_size, false);
    height_bufs_[0] = height;
    width_bufs_[0]  = width;

    if (controlnet_input_count > 0) {
        controlnet_res_bufs_[0] = (T*)allocator_->reMalloc(controlnet_res_bufs_[0], conv_buf_size, false);
    }

    // malloc hidden_states_bufs for down block res
    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        height_bufs_[i + 1] = (size_t)ceil(height_bufs_[i] / 2.0);
        width_bufs_[i + 1]  = (size_t)ceil(width_bufs_[i] / 2.0);

        size_t out1_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out2_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out3_size = sizeof(T) * batch_size * height_bufs_[i + 1] * width_bufs_[i + 1] * block_out_channels_[i];

        down_hidden_states_bufs_[i * 3] = (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3], out1_size, false);
        down_hidden_states_bufs_[i * 3 + 1] =
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], out2_size, false);
        down_hidden_states_bufs_[i * 3 + 2] =
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 2], out3_size, false);

        if (controlnet_input_count > 0) {
            controlnet_res_bufs_[i * 3 + 1] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 1], out1_size, false);
            controlnet_res_bufs_[i * 3 + 2] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 2], out2_size, false);
            controlnet_res_bufs_[i * 3 + 3] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 3], out3_size, false);
        }
    }

    size_t downblock_res_size = sizeof(T) * batch_size * height_bufs_[2] * width_bufs_[2] * block_out_channels_[2];
    down_hidden_states_bufs_[i * 3] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3], downblock_res_size, false);
    down_hidden_states_bufs_[i * 3 + 1] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], downblock_res_size, false);

    // malloc hidden_states_bufs for mid block res
    size_t mid_block_res_size = sizeof(T) * batch_size * height_bufs_[2] * width_bufs_[2] * block_out_channels_[2];
    mid_hidden_res_buf_       = (T*)allocator_->reMalloc(mid_hidden_res_buf_, mid_block_res_size, false);

    if (controlnet_input_count > 0) {
        controlnet_res_bufs_[i * 3 + 1] =
            (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 1], downblock_res_size, false);
        controlnet_res_bufs_[i * 3 + 2] =
            (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 2], downblock_res_size, false);
        controlnet_res_bufs_[i * 3 + 3] =
            (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 3], mid_block_res_size, false);
    }

    // malloc hidden_states_bufs for all up block res
    for (int j = block_out_channels_.size() - 1; j >= 0; j--) {
        size_t up_block_res_size =
            sizeof(T) * batch_size * height_bufs_[max(0, j - 1)] * width_bufs_[max(0, j - 1)] * block_out_channels_[j];
        up_hidden_states_bufs_[j] = (T*)allocator_->reMalloc(up_hidden_states_bufs_[j], up_block_res_size, false);
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void XLUnet2dConditionalModel<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&conv_buf_));
        allocator_->free((void**)(&temb_buf_));
        allocator_->free((void**)(&mid_hidden_res_buf_));
        allocator_->free((void**)(&norm_cache_buf_));
        if (use_runtime_augemb_) {
            allocator_->free((void**)(&time_proj_buf_));
        }

        for (int i = 0; i < down_hidden_states_bufs_.size(); i++) {
            allocator_->free((void**)(&down_hidden_states_bufs_[i]));
        }

        for (int i = 0; i < up_hidden_states_bufs_.size(); i++) {
            allocator_->free((void**)(&up_hidden_states_bufs_[i]));
        }

        if (controlnet_res_bufs_[0] != nullptr) {
            for (int i = 0; i < controlnet_res_bufs_.size(); i++) {
                allocator_->free((void**)(&controlnet_res_bufs_[i]));
            }
        }

        allocator_->freeAllNameBuf();
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void XLUnet2dConditionalModel<T>::forward(TensorMap*                                      output_tensors,
                                          const TensorMap*                                input_tensors,
                                          const float                                     timestep,
                                          const TensorMap*                                add_tensors,
                                          const XLUnet2dConditionalModelWeight<T>*        unet_weights,
                                          const std::vector<Tensor>*                      control_imgs,
                                          const std::vector<Tensor>*                      controlnet_augs,
                                          const std::vector<std::vector<float>>*          controlnet_scales,
                                          const std::vector<XLControlNetModelWeight<T>*>* controlnet_weights,
                                          const bool                                      controlnet_guess_mode)
{
    // input tensors:
    //      hidden_states: [bs, height, width, 4],
    //      tem: [bs, 1280]

    // output tensors:
    //      output_states_0: [bs, height, width, out_channels],

    if (control_imgs->size() != controlnet_scales->size()) {
        cout << "control_imgs size and controlnet_scales size not equal" << endl;
        throw "control_imgs size and controlnet_scales size not equal";
    }

    if (control_imgs->size() != controlnet_weights->size()) {
        cout << "control_imgs size and controlnet_weights size not equal" << endl;
        throw "control_imgs size and controlnet_weights size not equal";
    }

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor output                = output_tensors->at("output");

    size_t batch_size       = init_hidden_states.shape[0];
    size_t height           = init_hidden_states.shape[1];
    size_t width            = init_hidden_states.shape[2];
    size_t controlnet_count = control_imgs->size();

    LyraDiffContext* cur_context = input_tensors->context_;

    // 如果 height 和 width 一致，这里不需要再次 allocate
    allocateBuffer(batch_size, height, width, controlnet_count);

    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);

    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);
    Tensor              mid_hidden_states_res = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                                       {batch_size, height_bufs_[2], width_bufs_[2], block_out_channels_[2]},
                                          mid_hidden_res_buf_);
    std::vector<Tensor> down_hidden_states_res_vec;
    std::vector<Tensor> up_hidden_states_res_vec;
    std::vector<Tensor> controlnet_res_vec;

    controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height, width, block_out_channels_[0]},
                                        controlnet_res_bufs_[0]));

    // construct all down block tensor
    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3]));
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3 + 1]));
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i + 1], width_bufs_[i + 1], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3 + 2]));

        controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                            init_hidden_states.type,
                                            {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                            controlnet_res_bufs_[i * 3 + 1]));
        controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                            init_hidden_states.type,
                                            {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                            controlnet_res_bufs_[i * 3 + 2]));
        controlnet_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i + 1], width_bufs_[i + 1], block_out_channels_[i]},
                   controlnet_res_bufs_[i * 3 + 3]));
    }
    down_hidden_states_res_vec.push_back(Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                                down_hidden_states_bufs_[i * 3]));
    down_hidden_states_res_vec.push_back(Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                                down_hidden_states_bufs_[i * 3 + 1]));

    // downblock res
    controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                        controlnet_res_bufs_[i * 3 + 1]));
    controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                        controlnet_res_bufs_[i * 3 + 2]));
    // 这里是给到mid block res
    controlnet_res_vec.push_back(Tensor(MEMORY_GPU,
                                        init_hidden_states.type,
                                        {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                        controlnet_res_bufs_[i * 3 + 3]));

    if (down_hidden_states_res_vec.size() != 8) {
        throw "hidden_states_res_vec len wrong";
    }

    // construct all up block tensor
    for (int j = block_out_channels_.size() - 1; j >= 0; j--) {
        up_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[max(j - 1, 0)], width_bufs_[max(j - 1, 0)], block_out_channels_[j]},
                   up_hidden_states_bufs_[j]));
    }

    TensorMap output_tensor_map({{"output", temb}});

    // we can give either time_id in tensor or directly result emb.
    if (use_runtime_augemb_) {
        texttime_embedding->forward(&output_tensor_map, add_tensors, timestep, unet_weights->texttime_embedding_weight);
    }
    else {
        Tensor time_proj_tensor =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
        time_proj->forward(time_proj_tensor, timestep);
        // cout << "after time_prroj" << endl;
        timestep_embedding->forward(temb, time_proj_tensor, unet_weights->timestep_embedding_weight);
        invokeAddTensor2d(temb.getPtr<T>(),
                          temb.getPtr<T>(),
                          add_tensors->at("aug_emb").getPtr<T>(),
                          batch_size,
                          temb_channels_,
                          getStream());
        // cout << "after timestep_emb" << endl;
    }

    input_conv_->conv2dWithBias(conv_buf_,
                                // input_buf_,
                                init_hidden_states.getPtr<T>(),
                                unet_weights->conv_in_weight,
                                unet_weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    TensorMap input_tensor_map({{"hidden_states", conv_in_hidden_states}, {"temb", temb}});
    output_tensor_map = TensorMap({{"output_states_0", down_hidden_states_res_vec[0]},
                                   {"output_states_1", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    down_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->down_block_2d_weight);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                   {"round2_output", down_hidden_states_res_vec[4]},
                                   {"downsample_output", down_hidden_states_res_vec[5]}});
    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight1);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);
    output_tensor_map =
        TensorMap({{"round1_output", down_hidden_states_res_vec[6]}, {"round2_output", down_hidden_states_res_vec[7]}});
    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight2);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[7]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->mid_block_2d_weight);

    cur_context->is_controlnet = true;
    for (int k = 0; k < controlnet_count; k++) {
        TensorMap input_map = TensorMap({{"conditioning_img", control_imgs->at(k)},
                                         {"hidden_states", init_hidden_states},
                                         {"encoder_hidden_states", encoder_hidden_states}})
                                  .setContextThis(input_tensors);

        TensorMap add_input_map({{"aug_emb", controlnet_augs->at(k)}});

        controlnet->forward(controlnet_res_vec,
                            &input_map,
                            timestep,
                            &add_input_map,
                            controlnet_weights->at(k),
                            controlnet_scales->at(k));

        size_t cur_height  = conv_in_hidden_states.shape[1];
        size_t cur_width   = conv_in_hidden_states.shape[2];
        size_t cur_channel = conv_in_hidden_states.shape[3];

        if (controlnet_guess_mode) {
            for (int j = 0; j < controlnet_res_vec.size(); j++) {
                cudaMemsetAsync(
                    controlnet_res_vec[j].getPtr<T>(), 0, sizeof(T) * controlnet_res_vec[j].size() / 2, stream_);
            }
        }
        invokeAddResidual(conv_in_hidden_states.getPtr<T>(),
                          conv_in_hidden_states.getPtr<T>(),
                          controlnet_res_vec[0].getPtr<T>(),
                          1,
                          cur_channel,
                          cur_height,
                          cur_width,
                          batch_size,
                          stream_);
        int i = 0;

        for (; i < down_hidden_states_res_vec.size(); i++) {
            cur_height  = down_hidden_states_res_vec[i].shape[1];
            cur_width   = down_hidden_states_res_vec[i].shape[2];
            cur_channel = down_hidden_states_res_vec[i].shape[3];

            invokeAddResidual(down_hidden_states_res_vec[i].getPtr<T>(),
                              down_hidden_states_res_vec[i].getPtr<T>(),
                              controlnet_res_vec[i + 1].getPtr<T>(),
                              1,
                              cur_channel,
                              cur_height,
                              cur_width,
                              batch_size,
                              stream_);
        }

        cur_height  = mid_hidden_states_res.shape[1];
        cur_width   = mid_hidden_states_res.shape[2];
        cur_channel = mid_hidden_states_res.shape[3];

        invokeAddResidual(mid_hidden_states_res.getPtr<T>(),
                          mid_hidden_states_res.getPtr<T>(),
                          controlnet_res_vec[i + 1].getPtr<T>(),
                          1,
                          cur_channel,
                          cur_height,
                          cur_width,
                          batch_size,
                          stream_);
    }
    cur_context->is_controlnet = false;

    input_tensor_map = TensorMap({{"hidden_states", mid_hidden_states_res},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb},
                                  {"round1_input", down_hidden_states_res_vec[7]},
                                  {"round2_input", down_hidden_states_res_vec[6]},
                                  {"round3_input", down_hidden_states_res_vec[5]}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[0]}});
    cross_attn_up_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight1);

    input_tensor_map = TensorMap({{"hidden_states", up_hidden_states_res_vec[0]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb},
                                  {"round1_input", down_hidden_states_res_vec[4]},
                                  {"round2_input", down_hidden_states_res_vec[3]},
                                  {"round3_input", down_hidden_states_res_vec[2]}})
                           .setContextThis(input_tensors);
    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[1]}});

    cross_attn_up_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight2);

    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[1]},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[1]},
                                   {"round2_input", down_hidden_states_res_vec[0]},
                                   {"round3_input", conv_in_hidden_states}});
    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[2]}});

    up_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->up_block_2d_weight);

    invokeGroupNorm<T>(conv_buf_,
                       up_hidden_states_res_vec[2].getPtr<T>(),
                       unet_weights->conv_out_norm_gamma,
                       unet_weights->conv_out_norm_beta,
                       norm_cache_buf_,
                       batch_size,
                       height,
                       width,
                       block_out_channels_[0],
                       norm_num_groups_,
                       true,
                       getStream());

    output_conv_->conv2dWithBias(output.getPtr<T>(),
                                 //  input_buf_,
                                 conv_buf_,
                                 unet_weights->conv_out_weight,
                                 unet_weights->conv_out_bias,
                                 batch_size,
                                 height,
                                 width);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void XLUnet2dConditionalModel<T>::controlnet_forward(std::vector<Tensor>&              output_tensors,
                                                     const TensorMap*                  input_tensors,
                                                     const TensorMap*                  add_tensors,
                                                     const float                       timestep,
                                                     const XLControlNetModelWeight<T>* weights,
                                                     const std::vector<float>&         controlnet_scales)
{
    // 只是一个 wrapper，目的是让 controlnet 和 unet 可以共享显存
    controlnet->forward(output_tensors, input_tensors, timestep, add_tensors, weights, controlnet_scales);
}

template<typename T>
void XLUnet2dConditionalModel<T>::unet_forward(TensorMap*                               output_tensors,
                                               const TensorMap*                         input_tensors,
                                               const TensorMap*                         add_tensors,
                                               const float                              timestep,
                                               const XLUnet2dConditionalModelWeight<T>* unet_weights,
                                               const std::vector<Tensor>&               controlnet_results)
{
    // input tensors:
    //      hidden_states: [bs, height, width, 4],
    //      tem: [bs, 1280]

    // output tensors:
    //      output_states_0: [bs, height, width, out_channels],

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor output                = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    bool has_control_res = controlnet_results.size() > 0;

    LyraDiffContext* cur_context = input_tensors->context_;

    // 如果 height 和 width 一致，这里不需要再次 allocate
    allocateBuffer(batch_size, height, width, 0);

    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);

    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);

    Tensor mid_hidden_states_res = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                          {batch_size, height_bufs_[2], width_bufs_[2], block_out_channels_[2]},
                                          mid_hidden_res_buf_);

    std::vector<Tensor> down_hidden_states_res_vec;
    std::vector<Tensor> up_hidden_states_res_vec;

    // construct all down block tensor
    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3]));
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3 + 1]));
        down_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[i + 1], width_bufs_[i + 1], block_out_channels_[i]},
                   down_hidden_states_bufs_[i * 3 + 2]));
    }
    down_hidden_states_res_vec.push_back(Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                                down_hidden_states_bufs_[i * 3]));
    down_hidden_states_res_vec.push_back(Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]},
                                                down_hidden_states_bufs_[i * 3 + 1]));

    LYRA_CHECK_WITH_INFO(down_hidden_states_res_vec.size() == 8, "hidden_states_res_vec len wrong");

    // construct all up block tensor
    for (int j = block_out_channels_.size() - 1; j >= 0; j--) {
        up_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[max(j - 1, 0)], width_bufs_[max(j - 1, 0)], block_out_channels_[j]},
                   up_hidden_states_bufs_[j]));
    }

    TensorMap output_tensor_map({{"output", temb}});

    // we can give either time_id in tensor or directly result emb.
    if (use_runtime_augemb_) {
        texttime_embedding->forward(&output_tensor_map, add_tensors, timestep, unet_weights->texttime_embedding_weight);
    }
    else {
        Tensor time_proj_tensor =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
        time_proj->forward(time_proj_tensor, timestep);
        // cout << "after time_prroj" << endl;
        timestep_embedding->forward(temb, time_proj_tensor, unet_weights->timestep_embedding_weight);
        invokeAddTensor2d(temb.getPtr<T>(),
                          temb.getPtr<T>(),
                          add_tensors->at("aug_emb").getPtr<T>(),
                          batch_size,
                          temb_channels_,
                          getStream());
        // cout << "after timestep_emb" << endl;
    }

    input_conv_->conv2dWithBias(conv_buf_,
                                // input_buf_,
                                init_hidden_states.getPtr<T>(),
                                unet_weights->conv_in_weight,
                                unet_weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    TensorMap input_tensor_map({{"hidden_states", conv_in_hidden_states}, {"temb", temb}});
    output_tensor_map = TensorMap({{"output_states_0", down_hidden_states_res_vec[0]},
                                   {"output_states_1", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    down_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->down_block_2d_weight);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                   {"round2_output", down_hidden_states_res_vec[4]},
                                   {"downsample_output", down_hidden_states_res_vec[5]}});
    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight1);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);

    output_tensor_map =
        TensorMap({{"round1_output", down_hidden_states_res_vec[6]}, {"round2_output", down_hidden_states_res_vec[7]}});
    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight2);

    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[7]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->mid_block_2d_weight);

    // 如果有 controlnet res
    if (has_control_res) {
        LYRA_CHECK_WITH_INFO(controlnet_results.size() == 10, "controlnet_results len wrong");

        size_t cur_height  = conv_in_hidden_states.shape[1];
        size_t cur_width   = conv_in_hidden_states.shape[2];
        size_t cur_channel = conv_in_hidden_states.shape[3];

        invokeAddResidual(conv_in_hidden_states.getPtr<T>(),
                          conv_in_hidden_states.getPtr<T>(),
                          controlnet_results[0].getPtr<T>(),
                          1,
                          cur_channel,
                          cur_height,
                          cur_width,
                          batch_size,
                          stream_);
        int i = 0;

        for (; i < down_hidden_states_res_vec.size(); i++) {
            cur_height  = down_hidden_states_res_vec[i].shape[1];
            cur_width   = down_hidden_states_res_vec[i].shape[2];
            cur_channel = down_hidden_states_res_vec[i].shape[3];

            invokeAddResidual(down_hidden_states_res_vec[i].getPtr<T>(),
                              down_hidden_states_res_vec[i].getPtr<T>(),
                              controlnet_results[i + 1].getPtr<T>(),
                              1,
                              cur_channel,
                              cur_height,
                              cur_width,
                              batch_size,
                              stream_);
        }

        cur_height  = mid_hidden_states_res.shape[1];
        cur_width   = mid_hidden_states_res.shape[2];
        cur_channel = mid_hidden_states_res.shape[3];

        invokeAddResidual(mid_hidden_states_res.getPtr<T>(),
                          mid_hidden_states_res.getPtr<T>(),
                          controlnet_results[i + 1].getPtr<T>(),
                          1,
                          cur_channel,
                          cur_height,
                          cur_width,
                          batch_size,
                          stream_);
    }

    input_tensor_map = TensorMap({{"hidden_states", mid_hidden_states_res},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb},
                                  {"round1_input", down_hidden_states_res_vec[7]},
                                  {"round2_input", down_hidden_states_res_vec[6]},
                                  {"round3_input", down_hidden_states_res_vec[5]}})
                           .setContextThis(input_tensors);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[0]}});
    cross_attn_up_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight1);

    input_tensor_map = TensorMap({{"hidden_states", up_hidden_states_res_vec[0]},
                                  {"encoder_hidden_states", encoder_hidden_states},
                                  {"temb", temb},
                                  {"round1_input", down_hidden_states_res_vec[4]},
                                  {"round2_input", down_hidden_states_res_vec[3]},
                                  {"round3_input", down_hidden_states_res_vec[2]}})
                           .setContextThis(input_tensors);
    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[1]}});

    cross_attn_up_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight2);

    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[1]},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[1]},
                                   {"round2_input", down_hidden_states_res_vec[0]},
                                   {"round3_input", conv_in_hidden_states}});
    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[2]}});

    up_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->up_block_2d_weight);

    invokeGroupNorm<T>(conv_buf_,
                       up_hidden_states_res_vec[2].getPtr<T>(),
                       unet_weights->conv_out_norm_gamma,
                       unet_weights->conv_out_norm_beta,
                       norm_cache_buf_,
                       batch_size,
                       height,
                       width,
                       block_out_channels_[0],
                       norm_num_groups_,
                       true,
                       getStream());

    output_conv_->conv2dWithBias(output.getPtr<T>(),
                                 //  input_buf_,
                                 conv_buf_,
                                 unet_weights->conv_out_weight,
                                 unet_weights->conv_out_bias,
                                 batch_size,
                                 height,
                                 width);

    cublas_wrapper_->cublas_algo_map_->printAllShape();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
std::vector<std::vector<int64_t>>
XLUnet2dConditionalModel<T>::get_controlnet_results_shape(int64_t batch_size, int64_t height, int64_t width)
{
    std::vector<std::vector<int64_t>> res;

    height_bufs_[0] = height;
    width_bufs_[0]  = width;

    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        height_bufs_[i + 1] = (int64_t)ceil(height_bufs_[i] / 2.0);
        width_bufs_[i + 1]  = (int64_t)ceil(width_bufs_[i] / 2.0);
    }

    res.push_back({batch_size, height, width, block_out_channels_[0]});

    i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        height = (int64_t)ceil(height / 2.0);
        width  = (int64_t)ceil(width / 2.0);

        res.push_back({batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]});
        res.push_back({batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]});
        res.push_back({batch_size, height_bufs_[i + 1], width_bufs_[i + 1], block_out_channels_[i]});
    }

    res.push_back({batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]});
    res.push_back({batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]});
    res.push_back({batch_size, height_bufs_[i], width_bufs_[i], block_out_channels_[i]});

    return res;
}

template<typename T>
XLUnet2dConditionalModel<T>::~XLUnet2dConditionalModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;

    delete texttime_embedding;
    delete down_block_2d;
    delete cross_attn_down_block_2d_1;
    delete cross_attn_down_block_2d_2;
    delete mid_block_2d;
    delete cross_attn_up_block_2d_1;
    delete cross_attn_up_block_2d_2;
    delete up_block_2d;

    texttime_embedding         = nullptr;
    down_block_2d              = nullptr;
    cross_attn_down_block_2d_1 = nullptr;
    cross_attn_down_block_2d_2 = nullptr;
    mid_block_2d               = nullptr;
    cross_attn_up_block_2d_1   = nullptr;
    cross_attn_up_block_2d_2   = nullptr;
    up_block_2d                = nullptr;

    delete controlnet;
    controlnet = nullptr;

    freeBuffer();
}

template class XLUnet2dConditionalModel<float>;
template class XLUnet2dConditionalModel<half>;

}  // namespace lyradiff