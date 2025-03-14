#include "src/lyradiff/layers/upblock2d/XLUpBlock2d.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"

using namespace std;
namespace lyradiff {
template<typename T>
XLUpBlock2d<T>::XLUpBlock2d(const size_t     in_channels,
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

    if (in_channels_ != out_channels_) {
        throw "Upblock's in_channels_, out_channels_ should be the same";
    }

    resnet2d_block0 = new Resnet2DBlock<T>(out_channels_ + prev_output_channel_,
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

    resnet2d_block = new Resnet2DBlock<T>(out_channels_ + out_channels_,
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
}

template<typename T>
XLUpBlock2d<T>::XLUpBlock2d(XLUpBlock2d<T> const& up_block2d):
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

    resnet2d_block0 = up_block2d.resnet2d_block0;
    resnet2d_block  = up_block2d.resnet2d_block;
}

template<typename T>
XLUpBlock2d<T>::~XLUpBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete resnet2d_block0;
    delete resnet2d_block;
    freeBuffer();
}

template<typename T>
void XLUpBlock2d<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "XLUpBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void XLUpBlock2d<T>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t hidden_channel, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t cat_size0 = sizeof(T) * batch_size * height * width * (out_channels_ + hidden_channel);
    size_t cat_size  = sizeof(T) * batch_size * height * width * (out_channels_ + out_channels_);

    cat_buf_0_ = (T*)allocator_->reMallocWithName("XLUpBlock2d_cat_buf_0_", cat_size0, false);
    cat_buf_   = (T*)allocator_->reMallocWithName("XLUpBlock2d_cat_buf_", cat_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void XLUpBlock2d<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {

        allocator_->free((void**)(&cat_buf_0_));
        allocator_->free((void**)(&cat_buf_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void XLUpBlock2d<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                             const std::vector<lyradiff::Tensor>* input_tensors,
                             const XLUpBlock2dWeight<T>*        weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)},
                            {"temb", input_tensors->at(1)},
                            {"round1_input", input_tensors->at(2)},
                            {"round2_input", input_tensors->at(3)},
                            {"round3_input", input_tensors->at(4)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void XLUpBlock2d<T>::forward(TensorMap*                  output_tensors,
                             const TensorMap*            input_tensors,
                             const XLUpBlock2dWeight<T>* weights)
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

    allocateBuffer(batch_size, height, width, prev_output_channel_, target_height, target_width);

    invokeCatByChannel(cat_buf_0_,
                       init_hidden_states.getPtr<T>(),
                       round1_input.getPtr<T>(),
                       prev_output_channel_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 1
    Tensor    input_tensor0 = Tensor(MEMORY_GPU,
                                  init_hidden_states.type,
                                     {batch_size, height, width, out_channels_ + prev_output_channel_},
                                  cat_buf_0_);
    TensorMap input_map0    = TensorMap({{"hidden_states", input_tensor0}, {"temb", temb}});

    resnet2d_block0->forward(output_tensors, &input_map0, weights->resnet_2d_block_weight1);

    Tensor input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_ + out_channels_}, cat_buf_);
    TensorMap input_map = TensorMap({{"hidden_states", input_tensor}, {"temb", temb}});

    invokeCatByChannel(cat_buf_,
                       output.getPtr<T>(),
                       round2_input.getPtr<T>(),
                       out_channels_,
                       out_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 2
    resnet2d_block->forward(output_tensors, &input_map, weights->resnet_2d_block_weight2);
    invokeCatByChannel(cat_buf_,
                       output.getPtr<T>(),
                       round3_input.getPtr<T>(),
                       out_channels_,
                       in_channels_,
                       height,
                       width,
                       batch_size,
                       getStream());

    // round 3
    resnet2d_block->forward(output_tensors, &input_map, weights->resnet_2d_block_weight3);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class XLUpBlock2d<float>;
template class XLUpBlock2d<half>;
}  // namespace lyradiff
