#include "Unet2dConditionalModel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
// #include "src/lyradiff/layers/image_proj/ImageProjectBlock.h"
#include "cuda_runtime.h"
#include "src/lyradiff/utils/context.h"
using namespace std;

namespace lyradiff {

template<typename T>
Unet2dConditionalModel<T>::Unet2dConditionalModel(size_t max_controlnet_num,
                                                  //   std::vector<cudnnHandle_t> controlnet_cudnn_handles,
                                                  //   std::vector<cudaStream_t> controlnet_streams,
                                                  //   std::vector<cublasMMWrapper *> controlnet_cublas_wrappers,
                                                  //   std::vector<IAllocator *> controlnet_allocalors,
                                                  cudnnHandle_t    cudnn_handle,
                                                  cudaStream_t     stream,
                                                  cublasMMWrapper* cublas_wrapper,
                                                  IAllocator*      allocator,
                                                  const bool       is_free_buffer_after_forward,
                                                  const bool       sparse,
                                                  size_t           input_channels,
                                                  size_t           output_channels,
                                                  LyraQuantType    quant_level,
                                                  const std::string& sd_ver):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    cudnn_handle_(cudnn_handle),
    max_controlnet_num_(max_controlnet_num),
    input_channels_(input_channels),
    output_channels_(output_channels)
{
    if (sd_ver == "sd2")
        cross_attn_dim_ = 1024;
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }

    if (max_controlnet_num > 3) {
        throw "max_controlnet_num too big";
    }

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

    up_block_2d = new UpBlock2d<T>(block_out_channels_[2],
                                   block_out_channels_[3],
                                   block_out_channels_[3],
                                   norm_num_groups_,
                                   cudnn_handle_,
                                   stream,
                                   stream,
                                   cublas_wrapper,
                                   allocator,
                                   is_free_buffer_after_forward);

    cross_attn_up_block_2d_1 = new CrossAttnUpBlock2d<T, true>(block_out_channels_[1],
                                                               block_out_channels_[2],
                                                               block_out_channels_[3],
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

    cross_attn_up_block_2d_2 = new CrossAttnUpBlock2d<T, true>(block_out_channels_[0],
                                                               block_out_channels_[1],
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

    cross_attn_up_block_2d_3 = new CrossAttnUpBlock2d<T, false>(block_out_channels_[0],
                                                                block_out_channels_[0],
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

    controlnet = new ControlNetModel<T>(
        false, cudnn_handle, stream_, cublas_wrapper, allocator, is_free_buffer_after_forward, false, quant_level);

    // controlnet->timestep_embedding         = timestep_embedding;
    // controlnet->time_proj                  = time_proj;
    // controlnet->cross_attn_down_block_2d_1 = cross_attn_down_block_2d_1;
    // controlnet->cross_attn_down_block_2d_2 = cross_attn_down_block_2d_2;
    // controlnet->cross_attn_down_block_2d_3 = cross_attn_down_block_2d_3;
    // controlnet->down_block_2d              = down_block_2d;
    // controlnet->mid_block_2d               = mid_block_2d;
    // controlnet->input_conv_                = input_conv_;
}

template<typename T>
Unet2dConditionalModel<T>::Unet2dConditionalModel(Unet2dConditionalModel<T> const& unet):
    BaseLayer(unet.stream_,
              unet.cublas_wrapper_,
              unet.allocator_,
              unet.is_free_buffer_after_forward_,
              unet.cuda_device_prop_,
              unet.sparse_),
    cudnn_handle_(unet.cudnn_handle_)
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
    timestep_embedding         = unet.timestep_embedding;
    cross_attn_down_block_2d_1 = unet.cross_attn_down_block_2d_1;
    cross_attn_down_block_2d_2 = unet.cross_attn_down_block_2d_2;
    cross_attn_down_block_2d_3 = unet.cross_attn_down_block_2d_3;
    down_block_2d              = unet.down_block_2d;
    mid_block_2d               = unet.mid_block_2d;
    up_block_2d                = unet.up_block_2d;
    cross_attn_up_block_2d_1   = unet.cross_attn_up_block_2d_1;
    cross_attn_up_block_2d_2   = unet.cross_attn_up_block_2d_2;
    cross_attn_up_block_2d_3   = unet.cross_attn_up_block_2d_3;
    input_conv_                = unet.input_conv_;
    output_conv_               = unet.output_conv_;
    controlnet                 = unet.controlnet;
}

template<typename T>
void Unet2dConditionalModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "Unet2dConditionalModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void Unet2dConditionalModel<T>::allocateBuffer(size_t batch_size,
                                               size_t height,
                                               size_t width,
                                               size_t controlnet_input_count)
{
    cur_batch            = batch_size;
    cur_controlnet_count = controlnet_input_count;

    // size_t overall_size = 0;

    size_t conv_buf_size      = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t temb_buf_size      = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t norm_cache_size    = sizeof(double) * batch_size * norm_num_groups_ * 2;
    size_t time_proj_buf_size = sizeof(float) * batch_size * block_out_channels_[0];

    conv_buf_       = (T*)allocator_->reMalloc(conv_buf_, conv_buf_size, false);
    temb_buf_       = (T*)allocator_->reMalloc(temb_buf_, temb_buf_size, false);
    time_proj_buf_  = (T*)allocator_->reMalloc(time_proj_buf_, time_proj_buf_size, false);
    norm_cache_buf_ = (double*)allocator_->reMalloc(norm_cache_buf_, norm_cache_size, false);
    height_bufs_[0] = height;
    width_bufs_[0]  = width;

    if (controlnet_input_count > 0) {
        controlnet_res_bufs_[0] = (T*)allocator_->reMalloc(controlnet_res_bufs_[0], conv_buf_size, false);
    }

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
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 1], out2_size, false);
        down_hidden_states_bufs_[i * 3 + 2] =
            (T*)allocator_->reMalloc(down_hidden_states_bufs_[i * 3 + 2], out3_size, false);

        // overall_size += norm_cache_size;

        if (controlnet_input_count > 0) {
            controlnet_res_bufs_[i * 3 + 1] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 1], out1_size, false);
            controlnet_res_bufs_[i * 3 + 2] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 2], out2_size, false);
            controlnet_res_bufs_[i * 3 + 3] =
                (T*)allocator_->reMalloc(controlnet_res_bufs_[i * 3 + 3], out3_size, false);
        }
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
        // overall_size += up_block_res_size;
    }

    // cout << "unet allocate overall buf size " << overall_size / 1024 / 1024 << " MBs" << endl;

    is_allocate_buffer_ = true;
}

template<typename T>
void Unet2dConditionalModel<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&conv_buf_));
        allocator_->free((void**)(&temb_buf_));
        allocator_->free((void**)(&time_proj_buf_));
        allocator_->free((void**)(&mid_hidden_res_buf_));
        allocator_->free((void**)(&norm_cache_buf_));

        conv_buf_           = nullptr;
        temb_buf_           = nullptr;
        time_proj_buf_      = nullptr;
        mid_hidden_res_buf_ = nullptr;
        norm_cache_buf_     = nullptr;

        for (int i = 0; i < down_hidden_states_bufs_.size(); i++) {
            allocator_->free((void**)(&down_hidden_states_bufs_[i]));
            down_hidden_states_bufs_[i] = nullptr;
        }

        for (int i = 0; i < up_hidden_states_bufs_.size(); i++) {
            allocator_->free((void**)(&up_hidden_states_bufs_[i]));
            up_hidden_states_bufs_[i] = nullptr;
        }

        if (controlnet_res_bufs_[0] != nullptr) {
            for (int i = 0; i < controlnet_res_bufs_.size(); i++) {
                allocator_->free((void**)(&controlnet_res_bufs_[i]));
                controlnet_res_bufs_[i] = nullptr;
            }
        }

        allocator_->freeAllNameBuf();
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void Unet2dConditionalModel<T>::forward(TensorMap*                                    output_tensors,
                                        const TensorMap*                              input_tensors,
                                        const float                                   timestep,
                                        const Unet2dConditionalModelWeight<T>*        unet_weights,
                                        const std::vector<Tensor>*                    control_imgs,
                                        const std::vector<std::vector<float>>*        controlnet_scales,
                                        const std::vector<ControlNetModelWeight<T>*>* controlnet_weights,
                                        const bool                                    controlnet_guess_mode)
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

    if (control_imgs->size() > max_controlnet_num_) {
        cout << "control_imgs size greater than max_controlnet_num" << endl;
        throw "control_imgs size greater than max_controlnet_num";
    }

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor output                = output_tensors->at("output");

    size_t batch_size       = init_hidden_states.shape[0];
    size_t height           = init_hidden_states.shape[1];
    size_t width            = init_hidden_states.shape[2];
    size_t controlnet_count = control_imgs->size();

    // cout << "cur controlnet count " << controlnet_count << endl;

    // 如果 height 和 width 一致，这里不需要再次 allocate

    allocateBuffer(batch_size, height, width, controlnet_count);

    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);
    Tensor time_proj_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);
    Tensor              mid_hidden_states_res = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                                       {batch_size, height_bufs_[3], width_bufs_[3], block_out_channels_[3]},
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

    if (down_hidden_states_res_vec.size() != 11) {
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

    // cout << "after all tensors prepare" << endl;

    // temp
    TensorMap input_tensor_map({{"input", time_proj_tensor}});
    TensorMap output_tensor_map({{"output", time_proj_tensor}});

    // cudaEventRecord(start);
    time_proj->forward(&output_tensor_map, timestep);
    output_tensor_map = TensorMap({{"output", temb}});
    // timestep 和 input conv 可以并行，所以给到了不同的 stream
    timestep_embedding->forward(&output_tensor_map, &input_tensor_map, unet_weights->timestep_embedding_weight);

    // cout << "after timestep_embedding" << endl;
    input_conv_->conv2dWithBias(conv_buf_,
                                init_hidden_states.getPtr<T>(),
                                unet_weights->conv_in_weight,
                                unet_weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[0]},
                                   {"round2_output", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    input_tensor_map  = TensorMap({{"hidden_states", conv_in_hidden_states},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight1);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                   {"round2_output", down_hidden_states_res_vec[4]},
                                   {"downsample_output", down_hidden_states_res_vec[5]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight2);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[6]},
                                   {"round2_output", down_hidden_states_res_vec[7]},
                                   {"downsample_output", down_hidden_states_res_vec[8]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_3->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight3);

    output_tensor_map = TensorMap(
        {{"output_states_0", down_hidden_states_res_vec[9]}, {"output_states_1", down_hidden_states_res_vec[10]}});
    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[8]}, {"temb", temb}});

    down_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->down_block_2d_weight);

    output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[10]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->mid_block_2d_weight);

    for (int k = 0; k < controlnet_count; k++) {
        // cout << "cur controlnet idx " << k << endl;
        TensorMap input_map({{"conditioning_img", control_imgs->at(k)},
                             {"hidden_states", init_hidden_states},
                             {"encoder_hidden_states", encoder_hidden_states}});

        controlnet->forward(
            controlnet_res_vec, &input_map, timestep, controlnet_weights->at(k), controlnet_scales->at(k));

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

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[0]}});
    input_tensor_map  = TensorMap({{"hidden_states", mid_hidden_states_res},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[10]},
                                   {"round2_input", down_hidden_states_res_vec[9]},
                                   {"round3_input", down_hidden_states_res_vec[8]}})
                           .setContextThis(input_tensors);

    up_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->up_block_2d_weight);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[1]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[0]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[7]},
                                   {"round2_input", down_hidden_states_res_vec[6]},
                                   {"round3_input", down_hidden_states_res_vec[5]}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight1);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[2]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[1]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[4]},
                                   {"round2_input", down_hidden_states_res_vec[3]},
                                   {"round3_input", down_hidden_states_res_vec[2]}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight2);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[3]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[2]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[1]},
                                   {"round2_input", down_hidden_states_res_vec[0]},
                                   {"round3_input", conv_in_hidden_states}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_3->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight3);

    Tensor final_gnorm =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);

    invokeGroupNorm<T>(conv_buf_,
                       up_hidden_states_res_vec[3].getPtr<T>(),
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
void Unet2dConditionalModel<T>::controlnet_forward(std::vector<Tensor>&            output_tensors,
                                                   const TensorMap*                input_tensors,
                                                   const float                     timestep,
                                                   const ControlNetModelWeight<T>* weights,
                                                   const std::vector<float>&       controlnet_scales)
{
    // 只是一个 wrapper，目的是让 controlnet 和 unet 可以共享显存
    controlnet->forward(output_tensors, input_tensors, timestep, weights, controlnet_scales);
}

template<typename T>
void Unet2dConditionalModel<T>::unet_forward(TensorMap*                             output_tensors,
                                             const TensorMap*                       input_tensors,
                                             const float                            timestep,
                                             const Unet2dConditionalModelWeight<T>* unet_weights,
                                             const std::vector<Tensor>&             controlnet_results)
{
    input_tensors->context_->cur_running_module = "unet";
    cublas_wrapper_glob                         = this->cublas_wrapper_;
    Tensor init_hidden_states                   = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states                = input_tensors->at("encoder_hidden_states");
    Tensor output                               = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    bool has_control_res = controlnet_results.size() > 0;

    // 如果 height 和 width 一致，这里不需要再次 allocate
    allocateBuffer(batch_size, height, width, 0);

    Tensor temb = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, temb_channels_}, temb_buf_);
    Tensor time_proj_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
    Tensor conv_in_hidden_states =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);
    Tensor              mid_hidden_states_res = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                                       {batch_size, height_bufs_[3], width_bufs_[3], block_out_channels_[3]},
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

    LYRA_CHECK_WITH_INFO(down_hidden_states_res_vec.size() == 11, "hidden_states_res_vec len wrong");

    // construct all up block tensor
    for (int j = block_out_channels_.size() - 1; j >= 0; j--) {
        up_hidden_states_res_vec.push_back(
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, height_bufs_[max(j - 1, 0)], width_bufs_[max(j - 1, 0)], block_out_channels_[j]},
                   up_hidden_states_bufs_[j]));
    }

    // cout << "after all tensors prepare" << endl;

    TensorMap input_tensor_map({{"input", time_proj_tensor}});
    TensorMap output_tensor_map({{"output", time_proj_tensor}});

    time_proj->forward(&output_tensor_map, timestep);
    output_tensor_map = TensorMap({{"output", temb}});
    // timestep 和 input conv 可以并行，所以给到了不同的 stream
    timestep_embedding->forward(&output_tensor_map, &input_tensor_map, unet_weights->timestep_embedding_weight);

    // cout << "after timestep_embedding" << endl;
    input_conv_->conv2dWithBias(conv_buf_,
                                init_hidden_states.getPtr<T>(),
                                unet_weights->conv_in_weight,
                                unet_weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[0]},
                                   {"round2_output", down_hidden_states_res_vec[1]},
                                   {"downsample_output", down_hidden_states_res_vec[2]}});
    input_tensor_map  = TensorMap({{"hidden_states", conv_in_hidden_states},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight1);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[3]},
                                   {"round2_output", down_hidden_states_res_vec[4]},
                                   {"downsample_output", down_hidden_states_res_vec[5]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[2]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight2);

    output_tensor_map = TensorMap({{"round1_output", down_hidden_states_res_vec[6]},
                                   {"round2_output", down_hidden_states_res_vec[7]},
                                   {"downsample_output", down_hidden_states_res_vec[8]}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[5]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    cross_attn_down_block_2d_3->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_down_block_2d_weight3);

    output_tensor_map = TensorMap(
        {{"output_states_0", down_hidden_states_res_vec[9]}, {"output_states_1", down_hidden_states_res_vec[10]}});
    input_tensor_map = TensorMap({{"hidden_states", down_hidden_states_res_vec[8]}, {"temb", temb}});

    down_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->down_block_2d_weight);

    output_tensor_map = TensorMap({{"output", mid_hidden_states_res}});
    input_tensor_map  = TensorMap({{"hidden_states", down_hidden_states_res_vec[10]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb}})
                           .setContextThis(input_tensors);

    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->mid_block_2d_weight);

    // 如果有 controlnet res
    if (has_control_res) {
        LYRA_CHECK_WITH_INFO(controlnet_results.size() == 13, "controlnet_results len wrong");

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

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[0]}});
    input_tensor_map  = TensorMap({{"hidden_states", mid_hidden_states_res},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[10]},
                                   {"round2_input", down_hidden_states_res_vec[9]},
                                   {"round3_input", down_hidden_states_res_vec[8]}})
                           .setContextThis(input_tensors);

    up_block_2d->forward(&output_tensor_map, &input_tensor_map, unet_weights->up_block_2d_weight);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[1]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[0]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[7]},
                                   {"round2_input", down_hidden_states_res_vec[6]},
                                   {"round3_input", down_hidden_states_res_vec[5]}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight1);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[2]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[1]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[4]},
                                   {"round2_input", down_hidden_states_res_vec[3]},
                                   {"round3_input", down_hidden_states_res_vec[2]}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight2);

    output_tensor_map = TensorMap({{"output", up_hidden_states_res_vec[3]}});
    input_tensor_map  = TensorMap({{"hidden_states", up_hidden_states_res_vec[2]},
                                   {"encoder_hidden_states", encoder_hidden_states},
                                   {"temb", temb},
                                   {"round1_input", down_hidden_states_res_vec[1]},
                                   {"round2_input", down_hidden_states_res_vec[0]},
                                   {"round3_input", conv_in_hidden_states}})
                           .setContextThis(input_tensors);

    cross_attn_up_block_2d_3->forward(
        &output_tensor_map, &input_tensor_map, unet_weights->cross_attn_up_block_2d_weight3);

    Tensor final_gnorm =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, conv_buf_);

    invokeGroupNorm<T>(conv_buf_,
                       up_hidden_states_res_vec[3].getPtr<T>(),
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
                                 conv_buf_,
                                 unet_weights->conv_out_weight,
                                 unet_weights->conv_out_bias,
                                 batch_size,
                                 height,
                                 width);

    // cublas_wrapper_->cublas_algo_map_->printAllShape();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
std::vector<std::vector<int64_t>>
Unet2dConditionalModel<T>::get_controlnet_results_shape(int64_t batch_size, int64_t height, int64_t width)
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
Unet2dConditionalModel<T>::~Unet2dConditionalModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete timestep_embedding;
    delete time_proj;
    delete cross_attn_down_block_2d_1;
    delete cross_attn_down_block_2d_2;
    delete cross_attn_down_block_2d_3;
    delete down_block_2d;
    delete mid_block_2d;
    delete up_block_2d;
    delete cross_attn_up_block_2d_1;
    delete cross_attn_up_block_2d_2;
    delete cross_attn_up_block_2d_3;

    timestep_embedding         = nullptr;
    time_proj                  = nullptr;
    cross_attn_down_block_2d_1 = nullptr;
    cross_attn_down_block_2d_2 = nullptr;
    cross_attn_down_block_2d_3 = nullptr;
    down_block_2d              = nullptr;
    mid_block_2d               = nullptr;
    up_block_2d                = nullptr;
    cross_attn_up_block_2d_1   = nullptr;
    cross_attn_up_block_2d_2   = nullptr;
    cross_attn_up_block_2d_3   = nullptr;

    delete controlnet;
    controlnet = nullptr;

    freeBuffer();
}

template class Unet2dConditionalModel<float>;
template class Unet2dConditionalModel<half>;

}  // namespace lyradiff