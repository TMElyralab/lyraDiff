#include "XLControlNetModel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"

using namespace std;

namespace lyradiff {

template<typename T>
XLControlNetModel<T>::XLControlNetModel(bool                is_reuse_unet_blocks,
                                        cudnnHandle_t       cudnn_handle,
                                        cudaStream_t        stream,
                                        cublasMMWrapper*    cublas_wrapper,
                                        IAllocator*         allocator,
                                        const bool          is_free_buffer_after_forward,
                                        const bool          sparse,
                                        const bool          use_runtime_augemb,
                                        const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    cudnn_handle_(cudnn_handle),
    is_reuse_unet_blocks_(is_reuse_unet_blocks),
    use_runtime_augemb_(use_runtime_augemb)
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

    if (!is_reuse_unet_blocks_) {
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

            time_proj = new TimeProjection<T>(block_out_channels_[0],
                                              true,
                                              0,
                                              stream,
                                              cublas_wrapper,
                                              allocator,
                                              is_free_buffer_after_forward,
                                              false);

            timestep_embedding = new TimestepEmbeddingBlock<T>(block_out_channels_[0],
                                                               temb_channels_,
                                                               temb_channels_,
                                                               stream,
                                                               cublas_wrapper,
                                                               allocator,
                                                               is_free_buffer_after_forward,
                                                               false);
        }

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
                                    stream,
                                    cudnn_handle,
                                    allocator);

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

        small_down_block_2d_1 = new XLDownBlock2D<T>(block_out_channels_[0],
                                                     block_out_channels_[1],
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

        small_down_block_2d_2 = new XLDownBlock2D<T>(block_out_channels_[1],
                                                     block_out_channels_[2],
                                                     temb_channels_,
                                                     norm_num_groups_,
                                                     false,
                                                     false,
                                                     cudnn_handle_,
                                                     stream,
                                                     stream,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     false);

        small_mid_block_2d = new XLUNetMidBlock2DCrossAttn<T>(block_out_channels_[2],
                                                              temb_channels_,
                                                              norm_num_groups_,
                                                              head_nums_[1],
                                                              cross_attn_dim_,
                                                              0,
                                                              cudnn_handle_,
                                                              stream,
                                                              stream,
                                                              cublas_wrapper,
                                                              allocator,
                                                              is_free_buffer_after_forward,
                                                              false,
                                                              quant_level);
    }

    controlnet_conditioning_embedding = new ControlNetConditioningEmbedding<T>(controlnet_condition_channels_,
                                                                               block_out_channels_[0],
                                                                               cudnn_handle_,
                                                                               stream,
                                                                               cublas_wrapper,
                                                                               allocator,
                                                                               is_free_buffer_after_forward,
                                                                               false);

    controlnet_final_conv = new ControlNetFinalConv<T>({320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280},
                                                       cudnn_handle_,
                                                       stream,
                                                       cublas_wrapper,
                                                       allocator,
                                                       is_free_buffer_after_forward,
                                                       false);
}

template<typename T>
XLControlNetModel<T>::XLControlNetModel(XLControlNetModel<T> const& other):
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
    timestep_embedding                = other.timestep_embedding;
    cross_attn_down_block_2d_1        = other.cross_attn_down_block_2d_1;
    cross_attn_down_block_2d_2        = other.cross_attn_down_block_2d_2;
    down_block_2d                     = other.down_block_2d;
    small_down_block_2d_1             = other.small_down_block_2d_1;
    small_down_block_2d_2             = other.small_down_block_2d_2;
    small_mid_block_2d                = other.small_mid_block_2d;
    mid_block_2d                      = other.mid_block_2d;
    controlnet_conditioning_embedding = other.controlnet_conditioning_embedding;
    controlnet_final_conv             = other.controlnet_final_conv;
    input_conv_                       = other.input_conv_;
}

template<typename T>
void XLControlNetModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "XLControlNetModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void XLControlNetModel<T>::allocateBuffer(size_t batch_size, size_t hint_batch_size, size_t height, size_t width)
{
    cur_batch           = batch_size;
    cur_height          = height;
    cur_width           = width;
    cur_hint_batch_size = hint_batch_size;

    size_t conv_buf_size         = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t conditioning_buf_size = sizeof(T) * hint_batch_size * height * width * block_out_channels_[0];
    size_t temb_buf_size         = sizeof(T) * batch_size * temb_channels_;
    size_t time_proj_buf_size    = sizeof(float) * batch_size * block_out_channels_[0];

    conv_buf_         = (T*)allocator_->reMalloc(conv_buf_, conv_buf_size, false);
    conditioning_buf_ = (T*)allocator_->reMalloc(conditioning_buf_, conditioning_buf_size, false);
    temb_buf_         = (T*)allocator_->reMalloc(temb_buf_, temb_buf_size, false);
    time_proj_buf_    = (T*)allocator_->reMalloc(time_proj_buf_, time_proj_buf_size, false);
    height_bufs_[0]   = height;
    width_bufs_[0]    = width;

    // malloc hidden_states_bufs for cross attn down block res
    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        height_bufs_[i + 1] = (size_t)ceil(height_bufs_[i] / 2.0);
        width_bufs_[i + 1]  = (size_t)ceil(width_bufs_[i] / 2.0);

        size_t out1_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out2_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out3_size = sizeof(T) * batch_size * height_bufs_[i + 1] * width_bufs_[i + 1] * block_out_channels_[i];

        down_hidden_states_bufs_[i * 3] = (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3], out1_size, false);
        down_hidden_states_bufs_[i * 3 + 1] =
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], out1_size, false);
        down_hidden_states_bufs_[i * 3 + 2] =
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 2], out1_size, false);
    }
    // malloc hidden_states_bufs for down block res
    size_t downblock_res_size = sizeof(T) * batch_size * height_bufs_[2] * width_bufs_[2] * block_out_channels_[2];
    down_hidden_states_bufs_[i * 3] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3], downblock_res_size, false);
    down_hidden_states_bufs_[i * 3 + 1] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], downblock_res_size, false);

    // malloc hidden_states_bufs for mid block res
    size_t mid_block_res_size = sizeof(T) * batch_size * height_bufs_[2] * width_bufs_[2] * block_out_channels_[2];
    mid_hidden_res_buf_       = (T*)allocator_->reMalloc(mid_hidden_res_buf_, mid_block_res_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void XLControlNetModel<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&conv_buf_));
        allocator_->free((void**)(&temb_buf_));
        allocator_->free((void**)(&time_proj_buf_));
        allocator_->free((void**)(&mid_hidden_res_buf_));
        allocator_->free((void**)(&conditioning_buf_));

        for (int i = 0; i < down_hidden_states_bufs_.size(); i++) {
            allocator_->free((void**)(&down_hidden_states_bufs_[i]));
        }

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void XLControlNetModel<T>::forward(std::vector<Tensor>&              output_tensors,
                                   const TensorMap*                  input_tensors,
                                   const float                       timestep,
                                   const TensorMap*                  add_tensors,
                                   const XLControlNetModelWeight<T>* weights,
                                   const std::vector<float>&         controlnet_scales)
{

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor conditioning_img      = input_tensors->at("conditioning_img");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t hint_batch_size = conditioning_img.shape[0];

    // 如果 height 和 width 一致，这里不需要再次 allocate
    if (cur_height != height || cur_width != width || cur_batch != batch_size
        || cur_hint_batch_size != hint_batch_size) {
        allocateBuffer(batch_size, hint_batch_size, height, width);
    }

    // cout << "after allocateBuffer" << endl;

    // 提前准备所有Tensor
    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);
    Tensor time_proj_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);

    Tensor conditioning_cond_tensor = Tensor(MEMORY_GPU,
                                             init_hidden_states.type,
                                             {hint_batch_size, height, width, block_out_channels_[0]},
                                             conditioning_buf_);

    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);

    Tensor mid_hidden_states_res = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                          {batch_size, height_bufs_[2], width_bufs_[2], block_out_channels_[2]},
                                          mid_hidden_res_buf_);

    std::vector<Tensor> down_hidden_states_res_vec;

    // construct all down block tensor
    int i = 0;
    for (; i < block_out_channels_.size() - 1; i++) {
        size_t out1_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out2_size = sizeof(T) * batch_size * height_bufs_[i] * width_bufs_[i] * block_out_channels_[i];
        size_t out3_size = sizeof(T) * batch_size * height_bufs_[i + 1] * width_bufs_[i + 1] * block_out_channels_[i];

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

    if (down_hidden_states_res_vec.size() != 8) {
        throw "hidden_states_res_vec len wrong";
    }

    std::vector<Tensor> final_conv_input;
    final_conv_input.push_back(conv_in_hidden_states);
    for (int i = 0; i < down_hidden_states_res_vec.size(); i++) {
        final_conv_input.push_back(down_hidden_states_res_vec[i]);
    }
    final_conv_input.push_back(mid_hidden_states_res);

    // cout << "controlnet after input_conv_" << endl;

    // cout << "controlnet after invokeAddResidual" << endl;
    TensorMap output_tensor_map({{"output", temb}});

    // we can give either time_id in tensor or directly result emb.
    if (use_runtime_augemb_) {
        texttime_embedding->forward(&output_tensor_map, add_tensors, timestep, weights->texttime_embedding_weight);
    }
    else {
        Tensor time_proj_tensor =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
        time_proj->forward(time_proj_tensor, timestep);
        // cout << "after time_prroj" << endl;
        timestep_embedding->forward(temb, time_proj_tensor, weights->timestep_embedding_weight);
        invokeAddTensor2d(temb.getPtr<T>(),
                          temb.getPtr<T>(),
                          add_tensors->at("aug_emb").getPtr<T>(),
                          batch_size,
                          temb_channels_,
                          getStream());
        // cout << "after timestep_emb" << endl;
    }

    // cout << "after timestep_emb" << endl;

    // Tensor temb = Tensor::loadNpy("/home/mount/data/debug/emb.npy", MEMORY_GPU);
    // temb.saveNpy("/home/mount/data/debug/cpp_temb.npy");
    // printf("init hidden_states size: %d\n", init_hidden_states.size());
    input_conv_->conv2dWithBias(conv_buf_,
                                init_hidden_states.getPtr<T>(),
                                weights->conv_in_weight,
                                weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    // cout << "after input_conv_" << endl;

    output_tensor_map          = TensorMap({{"output", conditioning_cond_tensor}});
    TensorMap input_tensor_map = TensorMap({{"conditioning_img", conditioning_img}});

    controlnet_conditioning_embedding->forward(
        &output_tensor_map, &input_tensor_map, weights->controlnet_conditioning_embedding_weight);

    // cout << "controlnet after controlnet_conditioning_embedding" << endl;

    invokeAddResidualWithDifferentBatch(conv_buf_,
                                        conv_buf_,
                                        conditioning_buf_,
                                        1.0,
                                        block_out_channels_[0],
                                        height,
                                        width,
                                        batch_size,
                                        hint_batch_size,
                                        stream_);

    // cout << "after invokeAddResidualWithDifferentBatch" << endl;

    // Tensor conv_buf_t = Tensor(MEMORY_GPU, TYPE_FP16, {batch_size, height, width, block_out_channels_[0]},
    // conv_buf_); conv_buf_t.saveNpy("/home/mount/data/debug/cpp_conv_in.npy");

    // cout << "after input_conv_" << endl;
    input_tensor_map =
        TensorMap({{"hidden_states", conv_in_hidden_states}, {"temb", temb}}).setContextThis(input_tensors);
    output_tensor_map = TensorMap({{"output_states_0", down_hidden_states_res_vec[0]},
                                   {"output_states_1", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    down_block_2d->forward(&output_tensor_map, &input_tensor_map, weights->down_block_2d_weight);
    // output_tensor_map.at("downsample_output").saveNpy("/home/mount/data/debug/cpp_db0_out.npy");

    std::string controlnet_mode = weights->getControlnetMode();
    if (controlnet_mode == "large") {
        input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                      {"encoder_hidden_states", encoder_hidden_states},
                                      {"temb", temb}})
                               .setContextThis(input_tensors);
        output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                       {"round2_output", down_hidden_states_res_vec[4]},
                                       {"downsample_output", down_hidden_states_res_vec[5]}});
        cross_attn_down_block_2d_1->forward(
            &output_tensor_map, &input_tensor_map, weights->cross_attn_down_block_2d_weight1);

        // output_tensor_map.at("downsample_output").saveNpy("/home/mount/data/debug/cpp_db1_out.npy");

        input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                      {"encoder_hidden_states", encoder_hidden_states},
                                      {"temb", temb}})
                               .setContextThis(input_tensors);
        output_tensor_map = TensorMap(
            {{"round1_output", down_hidden_states_res_vec[6]}, {"round2_output", down_hidden_states_res_vec[7]}});
        cross_attn_down_block_2d_2->forward(
            &output_tensor_map, &input_tensor_map, weights->cross_attn_down_block_2d_weight2);

        // output_tensor_map.at("round2_output").saveNpy("/home/mount/data/debug/cpp_db2_out.npy");

        input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[7]},
                                      {"encoder_hidden_states", encoder_hidden_states},
                                      {"temb", temb}})
                               .setContextThis(input_tensors);
        output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
        mid_block_2d->forward(&output_tensor_map, &input_tensor_map, weights->mid_block_2d_weight);
    }
    else if (controlnet_mode == "small") {
        input_tensor_map =
            TensorMap({{"hidden_states", down_hidden_states_res_vec[2]}, {"temb", temb}}).setContextThis(input_tensors);
        output_tensor_map = TensorMap({{"output_states_0", down_hidden_states_res_vec[3]},
                                       {"output_states_1", down_hidden_states_res_vec[4]},
                                       {"downsample_output", down_hidden_states_res_vec[5]}});
        small_down_block_2d_1->forward(&output_tensor_map, &input_tensor_map, weights->small_down_block_2d_weight1);

        // output_tensor_map.at("downsample_output").saveNpy("/home/mount/data/debug/cpp_db1_out.npy");

        input_tensor_map =
            TensorMap({{"hidden_states", down_hidden_states_res_vec[5]}, {"temb", temb}}).setContextThis(input_tensors);
        output_tensor_map = TensorMap(
            {{"output_states_0", down_hidden_states_res_vec[6]}, {"output_states_1", down_hidden_states_res_vec[7]}});

        small_down_block_2d_2->forward(&output_tensor_map, &input_tensor_map, weights->small_down_block_2d_weight2);

        // output_tensor_map.at("round2_output").saveNpy("/home/mount/data/debug/cpp_db2_out.npy");

        input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[7]},
                                      {"encoder_hidden_states", encoder_hidden_states},
                                      {"temb", temb}})
                               .setContextThis(input_tensors);
        output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
        small_mid_block_2d->forward(&output_tensor_map, &input_tensor_map, weights->small_mid_block_2d_weight);
    }

    controlnet_final_conv->forward(
        output_tensors, final_conv_input, weights->controlnet_final_conv_weight, controlnet_scales);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
XLControlNetModel<T>::~XLControlNetModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;

    if (!is_reuse_unet_blocks_) {
        delete timestep_embedding;
        delete time_proj;
        delete cross_attn_down_block_2d_1;
        delete cross_attn_down_block_2d_2;
        delete down_block_2d;
        delete mid_block_2d;
        delete input_conv_;
        delete small_down_block_2d_1;
        delete small_down_block_2d_2;
        delete small_mid_block_2d;

        timestep_embedding         = nullptr;
        time_proj                  = nullptr;
        cross_attn_down_block_2d_1 = nullptr;
        cross_attn_down_block_2d_2 = nullptr;
        down_block_2d              = nullptr;
        mid_block_2d               = nullptr;
        input_conv_                = nullptr;
    }

    delete controlnet_conditioning_embedding;
    delete controlnet_final_conv;

    controlnet_conditioning_embedding = nullptr;
    controlnet_final_conv             = nullptr;

    freeBuffer();
}

template class XLControlNetModel<float>;
template class XLControlNetModel<half>;

}  // namespace lyradiff