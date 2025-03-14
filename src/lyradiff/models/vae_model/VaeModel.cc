#include "VaeModel.h"

using namespace std;

namespace lyradiff {

template<typename T>
VaeModel<T>::VaeModel(cudnnHandle_t    cudnn_handle,
                      cudaStream_t     stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator*      allocator,
                      const bool       is_free_buffer_after_forward,
                      const bool       sparse,
                      const bool       is_upcast):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    cudnn_handle_(cudnn_handle)
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

    post_quant_conv = new Conv2d<T>(latent_channels_,
                                    latent_channels_,
                                    1,
                                    1,
                                    0,
                                    0,
                                    CUDNN_TENSOR_NHWC,
                                    CUDNN_TENSOR_NHWC,
                                    CUDNN_TENSOR_NHWC,
                                    CUDNN_TENSOR_NHWC,
                                    stream_,
                                    cudnn_handle,
                                    allocator);

    quant_conv = new Conv2d<T>(latent_channels_ * 2,
                               latent_channels_ * 2,
                               1,
                               1,
                               0,
                               0,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_TENSOR_NHWC,
                               stream_,
                               cudnn_handle,
                               allocator);

    vae_decoder = new VaeDecoder<T>(latent_channels_,
                                    output_channels_,
                                    norm_num_groups_,
                                    cudnn_handle_,
                                    stream_,
                                    cublas_wrapper_,
                                    allocator_,
                                    is_free_buffer_after_forward,
                                    is_upcast);

    vae_encoder = new VaeEncoder<T>(input_channels_,
                                    latent_channels_,
                                    norm_num_groups_,
                                    cudnn_handle_,
                                    stream_,
                                    cublas_wrapper_,
                                    allocator_,
                                    is_free_buffer_after_forward);
}

template<typename T>
VaeModel<T>::VaeModel(VaeModel<T> const& other):
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
}

template<typename T>
void VaeModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "VaeModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void VaeModel<T>::decodeAllocateBuffer(size_t batch_size, size_t height, size_t width)
{
    prev_batch  = batch_size;
    prev_height = height;
    prev_width  = width;

    size_t post_quant_conv_buf_size = sizeof(T) * batch_size * height * width * latent_channels_;
    post_quant_conv_buf_            = (T*)allocator_->reMalloc(post_quant_conv_buf_, post_quant_conv_buf_size, false);

    size_t overall_size = 0;

    overall_size += post_quant_conv_buf_size;

    // cout << "cur vae model allocate overall buf size " << overall_size / 1024 / 1024 << " MBs" << endl;

    is_allocate_buffer_ = true;
}

template<typename T>
void VaeModel<T>::encodeAllocateBuffer(size_t batch_size, size_t height, size_t width)
{

    size_t encoder_res_buf_size = sizeof(T) * batch_size * height / 8 * width / 8 * latent_channels_ * 2;
    encoder_res_buf_            = (T*)allocator_->reMalloc(encoder_res_buf_, encoder_res_buf_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void VaeModel<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&post_quant_conv_buf_));
        allocator_->free((void**)(&encoder_res_buf_));
    }

    allocator_->freeAllNameBuf();
}

template<typename T>
void VaeModel<T>::decode(TensorMap*               output_tensors,
                         const TensorMap*         input_tensors,
                         const VaeModelWeight<T>* vae_weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, 4],

    // output tensors:
    //      output_states_0: [bs, height, width, out_channels],
    Tensor hidden_states = input_tensors->at("hidden_states");
    Tensor output        = output_tensors->at("output");

    size_t batch_size = hidden_states.shape[0];
    size_t height     = hidden_states.shape[1];
    size_t width      = hidden_states.shape[2];

    if (batch_size != prev_batch || height != prev_height || width != prev_width) {
        decodeAllocateBuffer(batch_size, height, width);
    }

    post_quant_conv->conv2dWithBias(post_quant_conv_buf_,
                                    hidden_states.getPtr<T>(),
                                    vae_weights->post_quant_conv_weight,
                                    vae_weights->post_quant_conv_bias,
                                    batch_size,
                                    height,
                                    width);

    Tensor decode_input_tensor =
        Tensor(MEMORY_GPU, hidden_states.type, {batch_size, height, width, latent_channels_}, post_quant_conv_buf_);

    // decode_input_tensor.saveNpy("/workspace/vae_model/post_quant_conv_res.npy");

    TensorMap input_map  = TensorMap({{"hidden_states", decode_input_tensor}});
    TensorMap output_map = TensorMap({{"output", output}});

    vae_decoder->forward(&output_map, &input_map, &vae_weights->vae_decoder_weight);

    cublas_wrapper_->cublas_algo_map_->printAllShape();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }

    // cudaDeviceSynchronize();
    // allocator_->printAllNameSize();
    // cout << "cur vae all remalloc size " << allocator_->getAllSize() / 1024 / 1024 << "MBs" << endl;
}

template<typename T>
void VaeModel<T>::encode(TensorMap*               output_tensors,
                         const TensorMap*         input_tensors,
                         const VaeModelWeight<T>* vae_weights)
{
    // input tensors:
    //      hidden_states: [bs, height, width, 3],

    // output tensors:
    //      output_states_0: [bs, height / 8, width / 8, latent_channels * 2],
    Tensor hidden_states = input_tensors->at("hidden_states");
    Tensor output        = output_tensors->at("output");

    size_t batch_size = hidden_states.shape[0];
    size_t height     = hidden_states.shape[1];
    size_t width      = hidden_states.shape[2];

    encodeAllocateBuffer(batch_size, height, width);

    Tensor encode_output_tensor = Tensor(
        MEMORY_GPU, hidden_states.type, {batch_size, height / 8, width / 8, latent_channels_ * 2}, encoder_res_buf_);

    // decode_input_tensor.saveNpy("/workspace/vae_model/post_quant_conv_res.npy");

    TensorMap input_map  = TensorMap({{"hidden_states", hidden_states}});
    TensorMap output_map = TensorMap({{"output", encode_output_tensor}});

    vae_encoder->forward(&output_map, &input_map, &vae_weights->vae_encoder_weight);

    // cout << "before quant_conv " << endl;

    quant_conv->conv2dWithBias(output.getPtr<T>(),
                               encoder_res_buf_,
                               vae_weights->quant_conv_weight,
                               vae_weights->quant_conv_bias,
                               batch_size,
                               height / 8,
                               width / 8);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }

    // cudaDeviceSynchronize();
    // allocator_->printAllNameSize();
    // cout << "cur vae all remalloc size " << allocator_->getAllSize() / 1024 / 1024 << "MBs" << endl;
}

template<typename T>
VaeModel<T>::~VaeModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;

    delete vae_decoder;

    vae_decoder = nullptr;

    freeBuffer();
}

template class VaeModel<float>;
template class VaeModel<half>;

}  // namespace lyradiff