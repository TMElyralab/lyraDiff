#include "GLVControlNetModel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"

using namespace std;

namespace lyradiff {

template<typename T>
GLVControlNetModel<T>::GLVControlNetModel(bool             is_reuse_unet_blocks,
                                          cudnnHandle_t    cudnn_handle,
                                          cudaStream_t     stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator*      allocator,
                                          const bool       is_free_buffer_after_forward,
                                          const bool       sparse,
                                          const bool       use_runtime_augemb):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
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
                                                                   is_free_buffer_after_forward);

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
                                                                   is_free_buffer_after_forward);

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
                                                        false);

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
}

template<typename T>
GLVControlNetModel<T>::GLVControlNetModel(GLVControlNetModel<T> const& other):
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
    timestep_embedding         = other.timestep_embedding;
    cross_attn_down_block_2d_1 = other.cross_attn_down_block_2d_1;
    cross_attn_down_block_2d_2 = other.cross_attn_down_block_2d_2;
    down_block_2d              = other.down_block_2d;
    mid_block_2d               = other.mid_block_2d;
    input_conv_                = other.input_conv_;
}

template<typename T>
void GLVControlNetModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "GLVControlNetModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void GLVControlNetModel<T>::allocateBuffer(size_t batch_size, size_t hint_batch_size, size_t height, size_t width)
{
    cur_batch           = batch_size;
    cur_height          = height;
    cur_width           = width;
    cur_hint_batch_size = hint_batch_size;

    size_t conditioning_buf_size = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t temb_buf_size         = sizeof(T) * batch_size * temb_channels_;
    size_t time_proj_buf_size    = sizeof(float) * batch_size * block_out_channels_[0];

    conditioning_buf_ = (T*)allocator_->reMalloc(conditioning_buf_, conditioning_buf_size, false);
    temb_buf_         = (T*)allocator_->reMalloc(temb_buf_, temb_buf_size, false);
    time_proj_buf_    = (T*)allocator_->reMalloc(time_proj_buf_, time_proj_buf_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void GLVControlNetModel<T>::freeBuffer()
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
void GLVControlNetModel<T>::forward(std::vector<Tensor>&               output_tensors,
                                    const TensorMap*                   input_tensors,
                                    const float                        timestep,
                                    const TensorMap*                   add_tensors,
                                    const GLVControlNetModelWeight<T>* weights)
{

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor img_condition         = input_tensors->at("conditioning_img");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t hint_batch_size = img_condition.shape[0];

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

    TensorMap output_tensor_map({{"output", temb}});

    // we can give either time_id in tensor or directly result emb.
    if (use_runtime_augemb_) {
        texttime_embedding->forward(&output_tensor_map, add_tensors, timestep, &weights->texttime_embedding_weight);
    }
    else {
        Tensor time_proj_tensor =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, block_out_channels_[0]}, time_proj_buf_);
        time_proj->forward(time_proj_tensor, timestep);
        // cout << "after time_prroj" << endl;
        timestep_embedding->forward(temb, time_proj_tensor, &weights->timestep_embedding_weight);
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
    input_conv_->conv2dWithBias(output_tensors[0].getPtr<T>(),
                                init_hidden_states.getPtr<T>(),
                                weights->conv_in_weight,
                                weights->conv_in_bias,
                                batch_size,
                                height,
                                width);

    input_conv_->conv2dWithBias(conditioning_buf_,
                                img_condition.getPtr<T>(),
                                weights->input_hint_conv_weight,
                                weights->input_hint_conv_bias,
                                batch_size,
                                height,
                                width);

    // cout << "controlnet after controlnet_conditioning_embedding" << endl;

    invokeAddResidualWithDifferentBatch(output_tensors[0].getPtr<T>(),
                                        output_tensors[0].getPtr<T>(),
                                        conditioning_buf_,
                                        1.0,
                                        block_out_channels_[0],
                                        height,
                                        width,
                                        batch_size,
                                        hint_batch_size,
                                        stream_);

    // cout << "after invokeAddResidualWithDifferentBatch" << endl;

    TensorMap input_tensor_map =
        TensorMap({{"hidden_states", output_tensors[0]}, {"temb", temb}});
    output_tensor_map = TensorMap({{"output_states_0", output_tensors[1]},
                                   {"output_states_1", output_tensors[2]},
                                   {"downsample_output", output_tensors[3]}});
    down_block_2d->forward(&output_tensor_map, &input_tensor_map, &weights->down_block_2d_weight);
    // output_tensor_map.at("downsample_output").saveNpy("/home/mount/data/debug/cpp_db0_out.npy");
    // cout << "after down_block_2d" << endl;

    input_tensor_map =
        TensorMap(
            {{"hidden_states", output_tensors[3]}, {"encoder_hidden_states", encoder_hidden_states}, {"temb", temb}});
    output_tensor_map = TensorMap({{"round1_output", output_tensors[4]},
                                   {"round2_output", output_tensors[5]},
                                   {"downsample_output", output_tensors[6]}});
    cross_attn_down_block_2d_1->forward(
        &output_tensor_map, &input_tensor_map, &weights->cross_attn_down_block_2d_weight1);

    // cout << "after cross_attn_down_block_2d_1" << endl;
    // output_tensor_map.at("downsample_output").saveNpy("/home/mount/data/debug/cpp_db1_out.npy");

    input_tensor_map =
        TensorMap(
            {{"hidden_states", output_tensors[6]}, {"encoder_hidden_states", encoder_hidden_states}, {"temb", temb}});
    output_tensor_map = TensorMap({{"round1_output", output_tensors[7]}, {"round2_output", output_tensors[8]}});
    cross_attn_down_block_2d_2->forward(
        &output_tensor_map, &input_tensor_map, &weights->cross_attn_down_block_2d_weight2);

    // output_tensor_map.at("round2_output").saveNpy("/home/mount/data/debug/cpp_db2_out.npy");

    // cout << "after cross_attn_down_block_2d_2" << endl;

    input_tensor_map =
        TensorMap(
            {{"hidden_states", output_tensors[8]}, {"encoder_hidden_states", encoder_hidden_states}, {"temb", temb}});
    output_tensor_map = TensorMap({{"output", output_tensors[9]}});
    mid_block_2d->forward(&output_tensor_map, &input_tensor_map, &weights->mid_block_2d_weight);

    // cout << "after mid_block_2d" << endl;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
GLVControlNetModel<T>::~GLVControlNetModel()
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

        timestep_embedding         = nullptr;
        time_proj                  = nullptr;
        cross_attn_down_block_2d_1 = nullptr;
        cross_attn_down_block_2d_2 = nullptr;
        down_block_2d              = nullptr;
        mid_block_2d               = nullptr;
        input_conv_                = nullptr;
    }

    freeBuffer();
}

template class GLVControlNetModel<float>;
template class GLVControlNetModel<half>;

}  // namespace lyradiff