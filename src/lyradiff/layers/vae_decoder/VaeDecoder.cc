#include "VaeDecoder.h"

using namespace std;
namespace lyradiff {
template<typename T>
VaeDecoder<T>::VaeDecoder(const size_t     in_channels,
                          const size_t     out_channels,
                          const size_t     norm_num_groups,
                          cudnnHandle_t    cudnn_handle,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward,
                          const bool       is_upcast):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
    in_channels_(in_channels),
    out_channels_(out_channels),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle),
    is_upcast_(is_upcast)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    if (std::is_same<T, float>()) {
        is_upcast_ = false;
    }

    conv_in = new Conv2d<T>(in_channels,
                            block_out_channels_[0],
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

    mid_block = new UNetMidBlock2D<T>(block_out_channels_[0],
                                      temb_channels_,
                                      norm_num_groups_,
                                      true,
                                      1,
                                      cudnn_handle_,
                                      stream_,
                                      cublas_wrapper,
                                      allocator,
                                      is_free_buffer_after_forward,
                                      false);

    up_decoder_block_0 = new UpDecoderBlock2d<T>(block_out_channels_[0],
                                                 block_out_channels_[0],
                                                 norm_num_groups_,
                                                 temb_channels_,
                                                 cudnn_handle_,
                                                 stream_,
                                                 cublas_wrapper,
                                                 allocator,
                                                 is_free_buffer_after_forward,
                                                 true,
                                                 is_upcast_);

    up_decoder_block_1 = new UpDecoderBlock2d<T>(block_out_channels_[0],
                                                 block_out_channels_[1],
                                                 norm_num_groups_,
                                                 temb_channels_,
                                                 cudnn_handle_,
                                                 stream_,
                                                 cublas_wrapper,
                                                 allocator,
                                                 is_free_buffer_after_forward,
                                                 true,
                                                 is_upcast_);

    if (is_upcast_) {
        upcast_up_decoder_block_2 = new UpDecoderBlock2d<float>(block_out_channels_[1],
                                                                block_out_channels_[2],
                                                                norm_num_groups_,
                                                                temb_channels_,
                                                                cudnn_handle_,
                                                                stream_,
                                                                cublas_wrapper,
                                                                allocator,
                                                                is_free_buffer_after_forward,
                                                                true,
                                                                is_upcast_);

        upcast_up_decoder_block_3 = new UpDecoderBlock2d<float>(block_out_channels_[2],
                                                                block_out_channels_[3],
                                                                norm_num_groups_,
                                                                temb_channels_,
                                                                cudnn_handle_,
                                                                stream_,
                                                                cublas_wrapper,
                                                                allocator,
                                                                is_free_buffer_after_forward,
                                                                false,
                                                                is_upcast_);
    }
    else {
        up_decoder_block_2 = new UpDecoderBlock2d<T>(block_out_channels_[1],
                                                     block_out_channels_[2],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     true,
                                                     is_upcast_);

        up_decoder_block_3 = new UpDecoderBlock2d<T>(block_out_channels_[2],
                                                     block_out_channels_[3],
                                                     norm_num_groups_,
                                                     temb_channels_,
                                                     cudnn_handle_,
                                                     stream_,
                                                     cublas_wrapper,
                                                     allocator,
                                                     is_free_buffer_after_forward,
                                                     false,
                                                     is_upcast_);
    }

    conv_out = new Conv2d<T>(block_out_channels_[3],
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
VaeDecoder<T>::VaeDecoder(VaeDecoder<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    in_channels_(other.in_channels_),
    out_channels_(other.out_channels_),
    norm_num_groups_(other.norm_num_groups_),
    cudnn_handle_(other.cudnn_handle_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    conv_in   = other.conv_in;
    conv_out  = other.conv_out;
    mid_block = other.mid_block;

    up_decoder_block_0 = other.up_decoder_block_0;
    up_decoder_block_1 = other.up_decoder_block_1;
    up_decoder_block_2 = other.up_decoder_block_2;
    up_decoder_block_3 = other.up_decoder_block_3;
}

template<typename T>
VaeDecoder<T>::~VaeDecoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete conv_in;
    delete conv_out;
    delete mid_block;
    delete up_decoder_block_0;
    delete up_decoder_block_1;
    if (is_upcast_) {
        delete upcast_up_decoder_block_2;
        delete upcast_up_decoder_block_3;
    }
    else {
        delete up_decoder_block_2;
        delete up_decoder_block_3;
    }

    freeBuffer();
}

template<typename T>
void VaeDecoder<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "VaeDecoder::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void VaeDecoder<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    prev_batch  = batch_size;
    prev_height = height;
    prev_width  = width;

    size_t mid_block_buf_size_ = sizeof(T) * batch_size * height * width * block_out_channels_[0];
    size_t gnorm_cache_size_   = sizeof(double) * batch_size * norm_num_groups_ * 2;

    if (is_upcast_) {
        size_t up_block_share_buf_size_ = sizeof(T) * batch_size * block_out_channels_[1] * height * 4 * width * 4;
        size_t upcast_up_block_share_buf_size_ =
            sizeof(float) * batch_size * block_out_channels_[2] * height * 8 * width * 8;
        size_t upcast_gnorm_buf_size_ = sizeof(float) * batch_size * height * 8 * width * 8 * block_out_channels_[3];

        up_block_share_buf_ =
            (T*)allocator_->reMallocWithName("VaeDecoder_up_block_share_buf_", up_block_share_buf_size_, false);
        upcast_up_block_share_buf_ = (float*)allocator_->reMallocWithName(
            "VaeDecoder_upcast_up_block_share_buf_", upcast_up_block_share_buf_size_, false);
        upcast_gnorm_buf_ =
            (float*)allocator_->reMallocWithName("VaeDecoder_upcast_gnorm_buf_", upcast_gnorm_buf_size_, false);
    }
    else {
        size_t up_block_share_buf_size_ = sizeof(T) * batch_size * block_out_channels_[2] * height * 8 * width * 8;
        size_t gnorm_buf_size_          = sizeof(T) * batch_size * height * 8 * width * 8 * block_out_channels_[3];

        up_block_share_buf_ =
            (T*)allocator_->reMallocWithName("VaeDecoder_up_block_share_buf_", up_block_share_buf_size_, false);
        gnorm_buf_ = (T*)allocator_->reMallocWithName("Resnet2DBlock_inner_conv_buf", gnorm_buf_size_, false);
    }

    mid_block_buf_ = (T*)allocator_->reMallocWithName("VaeDecoder_mid_block_buf_", mid_block_buf_size_, false);
    gnorm_cache_   = (double*)allocator_->reMallocWithName("VaeDecoder_gnorm_cache_", gnorm_cache_size_, false);

    // 因为upblock输入输出可以使用同一块显存，这里直接开辟一块最大的进行复用

    is_allocate_buffer_ = false;
}

template<typename T>
void VaeDecoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&up_block_share_buf_));
        allocator_->free((void**)(&mid_block_buf_));
        allocator_->free((void**)(&gnorm_buf_));
        allocator_->free((void**)(&gnorm_cache_));

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void VaeDecoder<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                            const std::vector<lyradiff::Tensor>* input_tensors,
                            const VaeDecoderWeight<T>*         weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void VaeDecoder<T>::forward(TensorMap*                 output_tensors,
                            const TensorMap*           input_tensors,
                            const VaeDecoderWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states = input_tensors->at("hidden_states");
    Tensor output             = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    allocateBuffer(batch_size, height, width);
    // cudaDeviceSynchronize();

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float elapsedTime;

    // cudaEventRecord(start);
    conv_in->conv2dWithBias(up_block_share_buf_,
                            init_hidden_states.getPtr<T>(),
                            weights->conv_in_weight,
                            weights->conv_in_bias,
                            batch_size,
                            height,
                            width);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae decode conv in cost: %f ms \n", elapsedTime);

    Tensor midblock_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, up_block_share_buf_);
    Tensor midblock_output_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, mid_block_buf_);

    TensorMap input_map  = TensorMap({{"hidden_states", midblock_input_tensor}});
    TensorMap output_map = TensorMap({{"output", midblock_output_tensor}});

    // cudaEventRecord(start);

    mid_block->forward(&output_map, &input_map, weights->unet_mid_block_2d_weight);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae decode mid_block cost: %f ms \n", elapsedTime);

    // midblock_output_tensor.saveNpy("/workspace/vae_model/midblock_output_tensor.npy");

    Tensor upblock_input_tensor = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, height, width, block_out_channels_[0]}, mid_block_buf_);
    Tensor upblock_output_tensor = Tensor(MEMORY_GPU,
                                          init_hidden_states.type,
                                          {batch_size, height * 2, width * 2, block_out_channels_[0]},
                                          up_block_share_buf_);

    input_map  = TensorMap({{"hidden_states", upblock_input_tensor}});
    output_map = TensorMap({{"output", upblock_output_tensor}});

    // cudaEventRecord(start);

    up_decoder_block_0->forward(&output_map, &input_map, weights->up_decoder_block_2d_weight_0);

    // upblock_output_tensor.saveNpy("/workspace/vae_model/upblock_output_tensor.npy");
    // cout << "up_decoder_block_0" << endl;

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae decode up_decoder_block_0 cost: %f ms \n", elapsedTime);

    Tensor upblock_output_tensor_1 = Tensor(MEMORY_GPU,
                                            init_hidden_states.type,
                                            {batch_size, height * 4, width * 4, block_out_channels_[1]},
                                            up_block_share_buf_);

    input_map  = TensorMap({{"hidden_states", upblock_output_tensor}});
    output_map = TensorMap({{"output", upblock_output_tensor_1}});

    // 这里可以复用up_decoder_block_0 因为0和1的in channel 以及out channel 一致
    // cudaEventRecord(start);

    up_decoder_block_0->forward(&output_map, &input_map, weights->up_decoder_block_2d_weight_1);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // printf("vae decode up_decoder_block_1 cost: %f ms \n", elapsedTime);

    // upblock_output_tensor_1.saveNpy("/workspace/vae_model/upblock_output_tensor_1.npy");
    // cout << "up_decoder_block_1" << endl;
    if (is_upcast_) {
        // cudaEventRecord(start);

        invokeCudaD2DcpyConvert(upcast_up_block_share_buf_,
                                up_block_share_buf_,
                                batch_size * height * 4 * width * 4 * block_out_channels_[1],
                                stream_);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode invokeCudaD2DcpyConvert cost: %f ms \n", elapsedTime);

        Tensor upblock_input_tensor_2 = Tensor(MEMORY_GPU,
                                               TYPE_FP32,
                                               {batch_size, height * 4, width * 4, block_out_channels_[1]},
                                               upcast_up_block_share_buf_);

        Tensor upblock_output_tensor_2 = Tensor(MEMORY_GPU,
                                                TYPE_FP32,
                                                {batch_size, height * 8, width * 8, block_out_channels_[2]},
                                                upcast_up_block_share_buf_);

        input_map  = TensorMap({{"hidden_states", upblock_input_tensor_2}});
        output_map = TensorMap({{"output", upblock_output_tensor_2}});

        // cudaEventRecord(start);

        upcast_up_decoder_block_2->forward(&output_map, &input_map, weights->upcast_up_decoder_block_2d_weight_2);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode upcast_up_decoder_block_2 cost: %f ms \n", elapsedTime);

        // upblock_output_tensor_2.saveNpy("/workspace/vae_model/upblock_output_tensor_2.npy");

        Tensor upblock_output_tensor_3 = Tensor(MEMORY_GPU,
                                                TYPE_FP32,
                                                {batch_size, height * 8, width * 8, block_out_channels_[3]},
                                                upcast_up_block_share_buf_);

        input_map  = TensorMap({{"hidden_states", upblock_output_tensor_2}});
        output_map = TensorMap({{"output", upblock_output_tensor_3}});

        // cudaEventRecord(start);

        upcast_up_decoder_block_3->forward(&output_map, &input_map, weights->upcast_up_decoder_block_2d_weight_3);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode upcast_up_decoder_block_3 cost: %f ms \n", elapsedTime);

        // upblock_output_tensor_3.saveNpy("/workspace/vae_model/up_decoder_block_2d_weight_3.npy");
        // cudaEventRecord(start);

        invokeGroupNorm(upcast_gnorm_buf_,
                        upcast_up_block_share_buf_,
                        weights->upcast_conv_norm_out_gamma,
                        weights->upcast_conv_norm_out_beta,
                        gnorm_cache_,
                        batch_size,
                        height * 8,
                        width * 8,
                        block_out_channels_[3],
                        norm_num_groups_,
                        true,
                        stream_);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode invokeGroupNorm cost: %f ms \n", elapsedTime);

        // cudaEventRecord(start);

        invokeCudaD2DcpyConvert(up_block_share_buf_,
                                upcast_gnorm_buf_,
                                batch_size * height * 8 * width * 8 * block_out_channels_[3],
                                stream_);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode invokeCudaD2DcpyConvert 2 cost: %f ms \n", elapsedTime);

        // cudaEventRecord(start);

        conv_out->conv2dWithBias(output.getPtr<T>(),
                                 up_block_share_buf_,
                                 weights->conv_out_weight,
                                 weights->conv_out_bias,
                                 batch_size,
                                 height * 8,
                                 width * 8);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);

        // printf("vae decode conv out cost: %f ms \n", elapsedTime);
    }
    else {
        Tensor upblock_output_tensor_2 = Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height * 8, width * 8, block_out_channels_[2]},
                                                up_block_share_buf_);

        input_map  = TensorMap({{"hidden_states", upblock_output_tensor_1}});
        output_map = TensorMap({{"output", upblock_output_tensor_2}});

        up_decoder_block_2->forward(&output_map, &input_map, weights->up_decoder_block_2d_weight_2);

        // upblock_output_tensor_2.saveNpy("/workspace/vae_model/upblock_output_tensor_2.npy");

        Tensor upblock_output_tensor_3 = Tensor(MEMORY_GPU,
                                                init_hidden_states.type,
                                                {batch_size, height * 8, width * 8, block_out_channels_[3]},
                                                up_block_share_buf_);

        input_map  = TensorMap({{"hidden_states", upblock_output_tensor_2}});
        output_map = TensorMap({{"output", upblock_output_tensor_3}});

        up_decoder_block_3->forward(&output_map, &input_map, weights->up_decoder_block_2d_weight_3);

        // upblock_output_tensor_3.saveNpy("/workspace/vae_model/up_decoder_block_2d_weight_3.npy");

        invokeGroupNorm(gnorm_buf_,
                        up_block_share_buf_,
                        weights->conv_norm_out_gamma,
                        weights->conv_norm_out_beta,
                        gnorm_cache_,
                        batch_size,
                        height * 8,
                        width * 8,
                        block_out_channels_[3],
                        norm_num_groups_,
                        true,
                        stream_);

        conv_out->conv2dWithBias(output.getPtr<T>(),
                                 gnorm_buf_,
                                 weights->conv_out_weight,
                                 weights->conv_out_bias,
                                 batch_size,
                                 height * 8,
                                 width * 8);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}
template class VaeDecoder<float>;
template class VaeDecoder<half>;
}  // namespace lyradiff