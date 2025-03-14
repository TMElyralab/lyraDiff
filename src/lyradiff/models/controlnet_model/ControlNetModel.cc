#include "ControlNetModel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"

using namespace std;

namespace lyradiff {

template<typename T>
ControlNetModel<T>::ControlNetModel(bool                is_reuse_unet_blocks,
                                    cudnnHandle_t       cudnn_handle,
                                    cudaStream_t        stream,
                                    cublasMMWrapper*    cublas_wrapper,
                                    IAllocator*         allocator,
                                    const bool          is_free_buffer_after_forward,
                                    const bool          sparse,
                                    const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    cudnn_handle_(cudnn_handle),
    is_reuse_unet_blocks_(is_reuse_unet_blocks)
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
        timestep_embedding = new TimestepEmbeddingBlock<T>(block_out_channels_[0],
                                                           temb_channels_,
                                                           temb_channels_,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           false);

        time_proj = new TimeProjection<T>(
            block_out_channels_[0], true, 0, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false);

        cross_attn_down_block_2d_1 = new CrossAttnDownBlock2d<T>(block_out_channels_[0],
                                                                 block_out_channels_[0],
                                                                 temb_channels_,
                                                                 head_num_,
                                                                 cross_attn_dim_,
                                                                 norm_num_groups_,
                                                                 cudnn_handle_,
                                                                 stream,
                                                                 stream,
                                                                 cublas_wrapper,
                                                                 allocator,
                                                                 is_free_buffer_after_forward,
                                                                 quant_level);

        cross_attn_down_block_2d_2 = new CrossAttnDownBlock2d<T>(block_out_channels_[0],
                                                                 block_out_channels_[1],
                                                                 temb_channels_,
                                                                 head_num_,
                                                                 cross_attn_dim_,
                                                                 norm_num_groups_,
                                                                 cudnn_handle_,
                                                                 stream,
                                                                 stream,
                                                                 cublas_wrapper,
                                                                 allocator,
                                                                 is_free_buffer_after_forward,
                                                                 quant_level);

        cross_attn_down_block_2d_3 = new CrossAttnDownBlock2d<T>(block_out_channels_[1],
                                                                 block_out_channels_[2],
                                                                 temb_channels_,
                                                                 head_num_,
                                                                 cross_attn_dim_,
                                                                 norm_num_groups_,
                                                                 cudnn_handle_,
                                                                 stream,
                                                                 stream,
                                                                 cublas_wrapper,
                                                                 allocator,
                                                                 is_free_buffer_after_forward,
                                                                 quant_level);

        down_block_2d = new DownBlock2D<T>(block_out_channels_[2],
                                           block_out_channels_[3],
                                           temb_channels_,
                                           norm_num_groups_,
                                           false,
                                           cudnn_handle_,
                                           stream,
                                           stream,
                                           cublas_wrapper,
                                           allocator,
                                           is_free_buffer_after_forward,
                                           false);

        mid_block_2d = new UNetMidBlock2DCrossAttn<T>(block_out_channels_[3],
                                                      temb_channels_,
                                                      norm_num_groups_,
                                                      head_num_,
                                                      cross_attn_dim_,
                                                      cudnn_handle_,
                                                      stream,
                                                      stream,
                                                      cublas_wrapper,
                                                      allocator,
                                                      is_free_buffer_after_forward,
                                                      false,
                                                      quant_level);

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
    }

    controlnet_conditioning_embedding = new ControlNetConditioningEmbedding<T>(controlnet_condition_channels_,
                                                                               block_out_channels_[0],
                                                                               cudnn_handle_,
                                                                               stream,
                                                                               cublas_wrapper,
                                                                               allocator,
                                                                               is_free_buffer_after_forward,
                                                                               false);

    controlnet_final_conv =
        new ControlNetFinalConv<T>({320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280},
                                   cudnn_handle_,
                                   stream,
                                   cublas_wrapper,
                                   allocator,
                                   is_free_buffer_after_forward,
                                   false);
}

template<typename T>
ControlNetModel<T>::ControlNetModel(ControlNetModel<T> const& other):
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
    cross_attn_down_block_2d_3        = other.cross_attn_down_block_2d_3;
    down_block_2d                     = other.down_block_2d;
    mid_block_2d                      = other.mid_block_2d;
    controlnet_conditioning_embedding = other.controlnet_conditioning_embedding;
    controlnet_final_conv             = other.controlnet_final_conv;
    input_conv_                       = other.input_conv_;
}

template<typename T>
void ControlNetModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "ControlNetModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void ControlNetModel<T>::allocateBuffer(size_t batch_size, size_t hint_batch_size, size_t height, size_t width)
{
    cur_batch = batch_size;

    size_t conv_buf_size         = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t conditioning_buf_size = sizeof(T) * hint_batch_size * height * width * block_out_channels_[0];
    size_t temb_buf_size         = sizeof(T) * batch_size * height * width * block_out_channels_[0];
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
    size_t downblock_res_size = sizeof(T) * batch_size * height_bufs_[3] * width_bufs_[3] * block_out_channels_[3];
    down_hidden_states_bufs_[i * 3] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3], downblock_res_size, false);
    down_hidden_states_bufs_[i * 3 + 1] =
        (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], downblock_res_size, false);

    // malloc hidden_states_bufs for mid block res
    size_t mid_block_res_size = sizeof(T) * batch_size * height_bufs_[3] * width_bufs_[3] * block_out_channels_[3];
    mid_hidden_res_buf_       = (T*)allocator_->reMalloc(mid_hidden_res_buf_, mid_block_res_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void ControlNetModel<T>::freeBuffer()
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
void ControlNetModel<T>::forward(std::vector<Tensor>&            output_tensors,
                                 const TensorMap*                input_tensors,
                                 const float                     timestep,
                                 const ControlNetModelWeight<T>* weights,
                                 const std::vector<float>&       controlnet_scales)
{

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor conditioning_img      = input_tensors->at("conditioning_img");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t hint_batch_size = conditioning_img.shape[0];

    // 如果 height 和 width 一致，这里不需要再次 allocate
    allocateBuffer(batch_size, hint_batch_size, height, width);

    // 提前准备所有Tensor
    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);
    Tensor time_proj_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);
    Tensor              conditioning_cond_tensor = Tensor(MEMORY_GPU,
                                             init_hidden_states.type,
                                                          {hint_batch_size, height, width, block_out_channels_[0]},
                                             conditioning_buf_);
    Tensor              mid_hidden_states_res    = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                                          {batch_size, height_bufs_[3], width_bufs_[3], block_out_channels_[3]},
                                          mid_hidden_res_buf_);
    std::vector<Tensor> down_hidden_states_res_vec;
    std::vector<Tensor> up_hidden_states_res_vec;

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

    if (down_hidden_states_res_vec.size() != 11) {
        throw "hidden_states_res_vec len wrong";
    }

    std::vector<Tensor> final_conv_input;
    final_conv_input.push_back(conv_in_hidden_states);
    for (int i = 0; i < down_hidden_states_res_vec.size(); i++) {
        final_conv_input.push_back(down_hidden_states_res_vec[i]);
    }
    final_conv_input.push_back(mid_hidden_states_res);

    // temp
    TensorMap input_tensor_map({{"input", time_proj_tensor}});
    TensorMap output_tensor_map({{"output", time_proj_tensor}});

    // cudaEventRecord(start);
    time_proj->forward(&output_tensor_map, timestep);
    output_tensor_map = TensorMap({{"output", temb}});
    // timestep 和 input conv 可以并行，所以给到了不同的 stream
    timestep_embedding->forward(&output_tensor_map, &input_tensor_map, weights->timestep_embedding_weight);

    // cout << "controlnet after timestep_embedding" << endl;
    input_conv_->conv2dWithBias(conv_buf_,
                                init_hidden_states.getPtr<T>(),
                                weights->conv_in_weight,
                                weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    // cout << "controlnet after input_conv_" << endl;

    output_tensor_map = TensorMap({{"output", conditioning_cond_tensor}});
    input_tensor_map  = TensorMap({{"conditioning_img", conditioning_img}});

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

    // cout << "controlnet after invokeAddResidual" << endl;

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[0]},
                                   {"round2_output", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    input_tensor_map  = TensorMap(
        {{"hidden_states", conv_in_hidden_states}, {"encoder_hidden_states", encoder_hidden_states}, {"temb", temb}});

    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, weights->cross_attn_down_block_2d_weight1);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                   {"round2_output", down_hidden_states_res_vec[4]},
                                   {"downsample_output", down_hidden_states_res_vec[5]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}});

    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, weights->cross_attn_down_block_2d_weight2);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[6]},
                                   {"round2_output", down_hidden_states_res_vec[7]},
                                   {"downsample_output", down_hidden_states_res_vec[8]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}});

    cross_attn_down_block_2d_3->forward(
        &output_tensor_map, &input_tensor_map, weights->cross_attn_down_block_2d_weight3);

    output_tensor_map = TensorMap(
        {{"output_states_0", down_hidden_states_res_vec[9]}, {"output_states_1", down_hidden_states_res_vec[10]}});
    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[8]}, {"temb", temb}});

    down_block_2d->forward(&output_tensor_map, &input_tensor_map, weights->down_block_2d_weight);

    output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[10]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}});

    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, weights->mid_block_2d_weight);

    controlnet_final_conv->forward(
        output_tensors, final_conv_input, weights->controlnet_final_conv_weight, controlnet_scales);

    // cout << "controlnet after controlnet_final_conv" << endl;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
ControlNetModel<T>::~ControlNetModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;

    if (!is_reuse_unet_blocks_) {
        delete timestep_embedding;
        delete time_proj;
        delete cross_attn_down_block_2d_1;
        delete cross_attn_down_block_2d_2;
        delete cross_attn_down_block_2d_3;
        delete down_block_2d;
        delete mid_block_2d;
        delete input_conv_;

        timestep_embedding         = nullptr;
        time_proj                  = nullptr;
        cross_attn_down_block_2d_1 = nullptr;
        cross_attn_down_block_2d_2 = nullptr;
        cross_attn_down_block_2d_3 = nullptr;
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

template class ControlNetModel<float>;
template class ControlNetModel<half>;

}  // namespace lyradiff