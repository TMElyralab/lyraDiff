#include "src/lyradiff/layers/transformer2d/Transformer2dBlock.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/norms/group_norm.h"

using namespace std;
namespace lyradiff {
template<typename T>
Transformer2dBlock<T>::Transformer2dBlock(size_t           in_channels,
                                          size_t           head_num,
                                          size_t           dim_per_head,
                                          size_t           cross_attn_dim,
                                          size_t           norm_num_groups,
                                          cudnnHandle_t    cudnn_handle,
                                          cudaStream_t     stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator*      allocator,
                                          bool             is_free_buffer_after_forward,
                                          LyraQuantType    quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false, quant_level),
    in_channels_(in_channels),
    head_num_(head_num),
    dim_per_head_(dim_per_head),
    inner_dim_(head_num * dim_per_head),
    cross_attn_dim_(cross_attn_dim),
    norm_num_groups_(norm_num_groups),
    cudnn_handle_(cudnn_handle)
{
    // sd2
    if (cross_attn_dim == 1024) {
        head_num_              = inner_dim_ / 64;
        dim_per_head_          = 64;
        use_linear_projection_ = true;
    }
    else {
        use_linear_projection_ = false;
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    // cout << "Transformer2dBlock in_channels_ " << in_channels_ << " head_num_ " << head_num_ << " dim_per_head_ "
    //      << dim_per_head_ << endl;

    if (quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block = new BasicTransformerInt8Block<T>(inner_dim_,
                                                                        head_num_,
                                                                        dim_per_head_,
                                                                        cross_attn_dim_,
                                                                        stream,
                                                                        cublas_wrapper,
                                                                        allocator,
                                                                        is_free_buffer_after_forward,
                                                                        quant_level);
    }
    else {
        basic_transformer_block = new BasicTransformerBlock<T>(inner_dim_,
                                                               head_num_,
                                                               dim_per_head_,
                                                               cross_attn_dim_,
                                                               stream,
                                                               cublas_wrapper,
                                                               allocator,
                                                               is_free_buffer_after_forward);
    }

    conv1 = new Conv2d<T>(in_channels_,
                          inner_dim_,
                          1,  // kernel size
                          1,
                          0,
                          0,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          stream,
                          cudnn_handle,
                          allocator);

    conv2 = new Conv2d<T>(inner_dim_,
                          in_channels_,
                          1,  // kernel size
                          1,
                          0,
                          0,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          CUDNN_TENSOR_NHWC,
                          stream,
                          cudnn_handle,
                          allocator);
}

template<typename T>
Transformer2dBlock<T>::Transformer2dBlock(Transformer2dBlock<T> const& transformer2dBlock):
    BaseLayer(transformer2dBlock.stream_,
              transformer2dBlock.cublas_wrapper_,
              transformer2dBlock.allocator_,
              transformer2dBlock.is_free_buffer_after_forward_,
              transformer2dBlock.cuda_device_prop_,
              transformer2dBlock.sparse_),
    in_channels_(transformer2dBlock.in_channels_),
    head_num_(transformer2dBlock.head_num_),
    dim_per_head_(transformer2dBlock.dim_per_head_),
    inner_dim_(transformer2dBlock.inner_dim_),
    cross_attn_dim_(transformer2dBlock.cross_attn_dim_),
    norm_num_groups_(transformer2dBlock.norm_num_groups_),
    cudnn_handle_(transformer2dBlock.cudnn_handle_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    basic_transformer_block      = transformer2dBlock.basic_transformer_block;
    basic_transformer_int8_block = transformer2dBlock.basic_transformer_int8_block;
}

template<typename T>
Transformer2dBlock<T>::~Transformer2dBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    if (quant_level_ != LyraQuantType::NONE) {
        delete basic_transformer_int8_block;
    }
    else {
        delete basic_transformer_block;
    }

    delete conv1;
    delete conv2;
    freeBuffer();
}

template<typename T>
void Transformer2dBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "Transformer2dBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void Transformer2dBlock<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size          = sizeof(T) * batch_size * height * width * in_channels_;
    size_t conv_out_size              = sizeof(T) * batch_size * height * width * inner_dim_;
    size_t basic_transformer_res_size = sizeof(T) * batch_size * height * width * inner_dim_;
    size_t norm_cache_size            = sizeof(double) * batch_size * norm_num_groups_ * 2;

    norm_hidden_state_buf_ =
        (T*)allocator_->reMallocWithName("Transformer2dBlock_norm_hidden_state_buf_", hidden_state_size, false);
    conv_out_buf_ = (T*)allocator_->reMallocWithName("Transformer2dBlock_conv_out_buf_", conv_out_size, false);
    basic_transformer_res_buf_ = (T*)allocator_->reMallocWithName(
        "Transformer2dBlock_basic_transformer_res_buf_", basic_transformer_res_size, false);
    norm_cache_buf_ =
        (double*)allocator_->reMallocWithName("Transformer2dBlock_norm_cache_buf_", norm_cache_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void Transformer2dBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&norm_hidden_state_buf_));
        allocator_->free((void**)(&conv_out_buf_));
        allocator_->free((void**)(&basic_transformer_res_buf_));
        allocator_->free((void**)(&norm_cache_buf_));

        is_allocate_buffer_ = false;
    }
}

/*
    hidden_states shape (N,H,W,C)
    encoder_hidden_states shape (N,77,768)
    output shape(N,H,W,C)
*/

template<typename T>
void Transformer2dBlock<T>::forward(std::vector<lyradiff::Tensor>*       output_tensors,
                                    const std::vector<lyradiff::Tensor>* input_tensors,
                                    const Transformer2dBlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}, {"encoder_hidden_states", input_tensors->at(1)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void Transformer2dBlock<T>::forward(TensorMap*                         output_tensors,
                                    const TensorMap*                   input_tensors,
                                    const Transformer2dBlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // cout << "Transformer2dBlock" << endl;

    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor output                = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    allocateBuffer(batch_size, height, width);

    // check_cuda_error(
    // 	cudaMemcpyAsync(output.getPtr<T>(), init_hidden_states.getPtr<T>(),
    // 					sizeof(T) * batch_size * height * width * in_channels_,
    // 					cudaMemcpyDeviceToDevice, getStream()));

    invokeGroupNorm<T>(norm_hidden_state_buf_,
                       init_hidden_states.getPtr<T>(),
                       weights->norm_gamma,
                       weights->norm_beta,
                       norm_cache_buf_,
                       batch_size,
                       height,
                       width,
                       in_channels_,
                       norm_num_groups_,
                       false,
                       getStream());

    // cout << "cur conv params, in channel: " << conv1->in_channels_ << " out channel: " << conv1->out_channels_ << "
    // kernel: " << conv1->kernel_size_ << " stride: " << conv1->stride_  << endl; cout << "cur conv input params, n: "
    // << batch_size << " h: " << height << " w: " << width << " c: " <<  conv1->in_channels_ << endl; cout << endl;
    if (!use_linear_projection_) {
        conv1->conv2dWithBias(conv_out_buf_,
                              norm_hidden_state_buf_,
                              weights->proj_in_weight,
                              weights->proj_in_bias,
                              batch_size,
                              height,
                              width);
    }
    else {
        int m = batch_size * height * width;
        int k = conv1->in_channels_;
        int n = conv1->out_channels_;
        // printf("[%d %d %d %d %d], [%d %d %d]\n",
        //     batch_size, height, width, conv1->in_channels_, conv1->out_channels_,
        //     m,n,k);
        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n,
                                                 m,
                                                 k,
                                                 weights->proj_in_weight,
                                                 k,
                                                 norm_hidden_state_buf_,
                                                 k,
                                                 conv_out_buf_,
                                                 n,
                                                 weights->proj_in_bias,
                                                 nullptr,
                                                 1.0f,
                                                 1.0f);
    }

    // datatype_enum datatype = TYPE_FP16;
    // if (std::is_same<T, float>())
    // {
    // 	datatype = TYPE_FP32;
    // }

    TensorMap input_tensor =
        TensorMap(
            {
                {"hidden_states",
                 Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height * width, inner_dim_}, conv_out_buf_)},
                {"encoder_hidden_states", encoder_hidden_states},
            })
            .setContextThis(input_tensors);

    TensorMap output_tensor({{"output",
                              Tensor(MEMORY_GPU,
                                     init_hidden_states.type,
                                     {batch_size, height * width, inner_dim_},
                                     basic_transformer_res_buf_)}});

    if (quant_level_ != LyraQuantType::NONE) {
        basic_transformer_int8_block->forward(
            &output_tensor, &input_tensor, weights->basic_transformer_int8_block_weight);
    }
    else {
        basic_transformer_block->forward(
            &output_tensor, &input_tensor, weights->basic_transformer_block_weight);
    }
    if (!use_linear_projection_) {
        conv2->conv2dWithBiasWithResidual(output.getPtr<T>(),
                                          basic_transformer_res_buf_,
                                          weights->proj_out_weight,
                                          weights->proj_out_bias,
                                          init_hidden_states.getPtr<T>(),
                                          batch_size,
                                          height,
                                          width,
                                          1.0f,
                                          1.0f);
    }
    else {
        int m = batch_size * height * width;
        int k = conv2->in_channels_;
        int n = conv2->out_channels_;
        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n,
                                                 m,
                                                 k,
                                                 weights->proj_out_weight,
                                                 k,
                                                 basic_transformer_res_buf_,
                                                 k,
                                                 output.getPtr<T>(),
                                                 n,
                                                 weights->proj_out_bias,
                                                 init_hidden_states.getPtr<T>(),
                                                 1.0f,
                                                 1.0f);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class Transformer2dBlock<float>;
template class Transformer2dBlock<half>;
}  // namespace lyradiff