#include "UpDecoderBlock2d.h"
#include "src/lyradiff/kernels/cross_attn_upblock_2d/cat_kernels.h"
#include "src/lyradiff/kernels/interpolate/interpolate.h"

using namespace std;
namespace lyradiff {
template<typename T>
UpDecoderBlock2d<T>::UpDecoderBlock2d(const size_t     in_channels,
                                      const size_t     out_channels,
                                      const size_t     norm_num_groups,
                                      const size_t     temb_channels,
                                      cudnnHandle_t    cudnn_handle,
                                      cudaStream_t     stream,
                                      cublasMMWrapper* cublas_wrapper,
                                      IAllocator*      allocator,
                                      bool             is_free_buffer_after_forward,
                                      bool             add_upsample,
                                      bool             is_upcast):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    temb_channels_(temb_channels),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle),
    add_upsample_(add_upsample),
    is_upcast_(is_upcast)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    resnet2d_block = new Resnet2DBlock<T>(out_channels_,
                                          out_channels,
                                          norm_num_groups_,
                                          norm_num_groups_,
                                          true,
                                          temb_channels_,
                                          cudnn_handle_,
                                          stream_,
                                          stream_,
                                          cublas_wrapper,
                                          allocator,
                                          is_free_buffer_after_forward,
                                          temb_channels_ > 0);

    resnet2d_block_pre = new Resnet2DBlock<T>(in_channels,
                                              out_channels,
                                              norm_num_groups_,
                                              norm_num_groups_,
                                              true,
                                              temb_channels_,
                                              cudnn_handle_,
                                              stream_,
                                              stream_,
                                              cublas_wrapper,
                                              allocator,
                                              is_free_buffer_after_forward,
                                              temb_channels_ > 0);

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
UpDecoderBlock2d<T>::UpDecoderBlock2d(UpDecoderBlock2d<T> const& up_decoder_block2d):
    BaseLayer(up_decoder_block2d.stream_,
              up_decoder_block2d.cublas_wrapper_,
              up_decoder_block2d.allocator_,
              up_decoder_block2d.is_free_buffer_after_forward_,
              up_decoder_block2d.cuda_device_prop_,
              up_decoder_block2d.sparse_),
    in_channels_(up_decoder_block2d.in_channels_),
    out_channels_(up_decoder_block2d.out_channels_),
    temb_channels_(up_decoder_block2d.temb_channels_),
    norm_num_groups_(up_decoder_block2d.norm_num_groups_),
    cudnn_handle_(up_decoder_block2d.cudnn_handle_),
    add_upsample_(up_decoder_block2d.add_upsample_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    resnet2d_block     = up_decoder_block2d.resnet2d_block;
    resnet2d_block_pre = up_decoder_block2d.resnet2d_block_pre;
    upsampler_conv     = up_decoder_block2d.upsampler_conv;
    is_upcast_         = up_decoder_block2d.is_upcast_;
}

template<typename T>
UpDecoderBlock2d<T>::~UpDecoderBlock2d()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete resnet2d_block;
    delete resnet2d_block_pre;
    delete upsampler_conv;
    freeBuffer();
}

template<typename T>
void UpDecoderBlock2d<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "UpDecoderBlock2d::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void UpDecoderBlock2d<T>::allocateBuffer(
    size_t batch_size, size_t height, size_t width, size_t target_height, size_t target_width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size = sizeof(T) * batch_size * height * width * out_channels_;
    size_t interpolate_size  = sizeof(T) * batch_size * target_height * target_width * out_channels_;

    if (is_upcast_) {
        hidden_state_buf_ = (T*)allocator_->reMallocWithName("VaeDecoder_upcast_gnorm_buf_", hidden_state_size, false);
    }
    else {
        hidden_state_buf_ =
            (T*)allocator_->reMallocWithName("UpDecoderBlock2d_hidden_state_buf_", hidden_state_size, false);
    }

    hidden_state_buf2_ =
        (T*)allocator_->reMallocWithName("UpDecoderBlock2d_hidden_state_buf2_", hidden_state_size, false);
    if (add_upsample_) {
        interpolate_buf_ = (T*)allocator_->reMallocWithName("Resnet2DBlock_inner_buf_1", interpolate_size, false);
        // hidden_state_buf2_ = interpolate_buf_;
    }

    is_allocate_buffer_ = false;
}

template<typename T>
void UpDecoderBlock2d<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&hidden_state_buf_));
        allocator_->free((void**)(&hidden_state_buf2_));
        if (add_upsample_) {
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

template<typename T>
void UpDecoderBlock2d<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                                  const std::vector<lyradiff::Tensor>* input_tensors,
                                  const UpDecoderBlock2dWeight<T>*   weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void UpDecoderBlock2d<T>::forward(TensorMap*                       output_tensors,
                                  const TensorMap*                 input_tensors,
                                  const UpDecoderBlock2dWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    size_t target_height = output.shape[1];
    size_t target_width  = output.shape[2];

    allocateBuffer(batch_size, height, width, target_height, target_width);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float elapsedTime;

    // cudaEventRecord(start);

    // round 1
    Tensor tensor1 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf_);
    Tensor tensor2 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, out_channels_}, hidden_state_buf2_);

    TensorMap input_tensor_map1 = TensorMap({{"hidden_states", tensor1}});
    TensorMap input_tensor_map2 = TensorMap({{"hidden_states", tensor2}});

    TensorMap output_tensor_map1 = TensorMap({{"output", tensor1}});
    TensorMap output_tensor_map2 = TensorMap({{"output", tensor2}});

    if (in_channels_ == out_channels_) {
        resnet2d_block->forward(&output_tensor_map1, input_tensors, weights->resnet_2d_block_weight1);
    }
    else {
        resnet2d_block_pre->forward(&output_tensor_map1, input_tensors, weights->resnet_2d_block_weight1);
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae UpDecoderBlock2d resnet 1 cost: %f ms \n", elapsedTime);

    // // tensor1.saveNpy("/workspace/vae_model/UpDecoderResnet_0.npy");
    // cudaEventRecord(start);

    // round 2
    resnet2d_block->forward(&output_tensor_map2, &input_tensor_map1, weights->resnet_2d_block_weight2);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae UpDecoderBlock2d resnet 2 cost: %f ms \n", elapsedTime);

    // round 3
    if (!add_upsample_) {
        // cudaEventRecord(start);

        resnet2d_block->forward(output_tensors, &input_tensor_map2, weights->resnet_2d_block_weight3);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae UpDecoderBlock2d no add_upsample_ resnet 3 cost: %f ms \n", elapsedTime);

        if (is_free_buffer_after_forward_ == true) {
            freeBuffer();
        }
        return;
    }

    // cudaEventRecord(start);

    resnet2d_block->forward(&output_tensor_map1, &input_tensor_map2, weights->resnet_2d_block_weight3);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae UpDecoderBlock2d resnet 3 cost: %f ms \n", elapsedTime);

    // invokeInterpolateNearest(interpolate_buf_, output_tensor.getPtr<T>(), batch_size, height, width, out_channels_,
    // 2, getStream());
    // cudaEventRecord(start);

    invokeInterpolateNearestToShape(interpolate_buf_,
                                    hidden_state_buf_,
                                    batch_size,
                                    height,
                                    width,
                                    out_channels_,
                                    target_height,
                                    target_width,
                                    getStream());

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae UpDecoderBlock2d invokeInterpolateNearestToShape cost: %f ms \n", elapsedTime);

    // cout << "cur conv params, in channel: " << upsampler_conv->in_channels_ << " out channel: " <<
    // upsampler_conv->out_channels_ << " kernel: " << upsampler_conv->kernel_size_ << " stride: " <<
    // upsampler_conv->stride_  << endl; cout << "cur conv input params, n: " << batch_size << " h: " << target_height
    // << " w: " << target_width << " c: " <<  upsampler_conv->in_channels_ << endl; cout << endl;
    // cudaEventRecord(start);

    upsampler_conv->conv2dWithBias(output.getPtr<T>(),
                                   interpolate_buf_,
                                   weights->upsampler_weight,
                                   weights->upsampler_bias,
                                   batch_size,
                                   target_height,
                                   target_width);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae UpDecoderBlock2d upsampler_conv cost: %f ms \n", elapsedTime);


    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class UpDecoderBlock2d<float>;
template class UpDecoderBlock2d<half>;
}  // namespace lyradiff