#include "src/lyradiff/layers/transformer2d/XLTransformer2dBlock.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/kernels/mid_block_2d/add_bias.h"
#include "src/lyradiff/kernels/norms/group_norm.h"
#include "src/lyradiff/kernels/timestep_embedding/ffn_kernels.h"

using namespace std;
namespace lyradiff {
template<typename T>
XLTransformer2dBlock<T>::XLTransformer2dBlock(size_t           in_channels,
                                              size_t           head_num,
                                              size_t           dim_per_head,
                                              size_t           cross_attn_dim,
                                              size_t           norm_num_groups,
                                              size_t           inner_trans_num,
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
    inner_trans_num_(inner_trans_num),
    cudnn_handle_(cudnn_handle)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (typeid(T) == typeid(half)) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else {
        cublas_wrapper_->setFP32GemmConfig();
    }

    if (quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < inner_trans_num; i++) {
            transblock_int8_vec.emplace_back(new BasicTransformerInt8Block<T>(inner_dim_,
                                                                              head_num_,
                                                                              dim_per_head_,
                                                                              cross_attn_dim_,
                                                                              stream,
                                                                              cublas_wrapper,
                                                                              allocator,
                                                                              is_free_buffer_after_forward,
                                                                              quant_level_));
        }
    }
    else {
        for (int i = 0; i < inner_trans_num; i++) {
            transblock_vec.emplace_back(new BasicTransformerBlock<T>(inner_dim_,
                                                                     head_num_,
                                                                     dim_per_head_,
                                                                     cross_attn_dim_,
                                                                     stream,
                                                                     cublas_wrapper,
                                                                     allocator,
                                                                     is_free_buffer_after_forward));
        }
    }
}

template<typename T>
XLTransformer2dBlock<T>::XLTransformer2dBlock(XLTransformer2dBlock<T> const& xltransformer2dBlock):
    BaseLayer(xltransformer2dBlock.stream_,
              xltransformer2dBlock.cublas_wrapper_,
              xltransformer2dBlock.allocator_,
              xltransformer2dBlock.is_free_buffer_after_forward_,
              xltransformer2dBlock.cuda_device_prop_,
              xltransformer2dBlock.sparse_),
    in_channels_(xltransformer2dBlock.in_channels_),
    head_num_(xltransformer2dBlock.head_num_),
    dim_per_head_(xltransformer2dBlock.dim_per_head_),
    inner_dim_(xltransformer2dBlock.inner_dim_),
    cross_attn_dim_(xltransformer2dBlock.cross_attn_dim_),
    norm_num_groups_(xltransformer2dBlock.norm_num_groups_),
    inner_trans_num_(xltransformer2dBlock.inner_trans_num_),
    cudnn_handle_(xltransformer2dBlock.cudnn_handle_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }
    // Do we need to reconstruct whole object? or new initiliazed xltransformer2dBlock will lose all it's inner
    // transblocks if the source object is destructed.
    transblock_vec      = xltransformer2dBlock.transblock_vec;
    transblock_int8_vec = xltransformer2dBlock.transblock_int8_vec;
}

template<typename T>
XLTransformer2dBlock<T>::~XLTransformer2dBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    // delete basic_transformer_block;
    // basic_transformer_block = nullptr;
    if (quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < transblock_int8_vec.size(); i++) {
            delete transblock_int8_vec[i];
        }
    }
    else {
        for (int i = 0; i < transblock_vec.size(); i++) {
            delete transblock_vec[i];
        }
    }

    freeBuffer();
}

template<typename T>
void XLTransformer2dBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "XLTransformer2dBlock::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void XLTransformer2dBlock<T>::allocateBuffer(size_t batch_size, size_t height, size_t width)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_state_size          = sizeof(T) * batch_size * height * width * in_channels_;
    size_t linear_out_size            = sizeof(T) * batch_size * height * width * inner_dim_;
    size_t basic_transformer_res_size = sizeof(T) * batch_size * height * width * inner_dim_;
    size_t norm_cache_size            = sizeof(double) * batch_size * norm_num_groups_ * 2;

    norm_hidden_state_buf_ =
        (T*)allocator_->reMallocWithName("XLTransformer2dBlock_norm_hidden_state_buf_", hidden_state_size, false);
    linear_out_buf_ = (T*)allocator_->reMallocWithName("XLTransformer2dBlock_linear_out_buf_", linear_out_size, false);
    basic_transformer_res_buf_ = (T*)allocator_->reMallocWithName(
        "XLTransformer2dBlock_basic_transformer_res_buf_", basic_transformer_res_size, false);
    norm_cache_buf_ =
        (double*)allocator_->reMallocWithName("XLTransformer2dBlock_norm_cache_buf_", norm_cache_size, false);

    is_allocate_buffer_ = false;
}

template<typename T>
void XLTransformer2dBlock<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&norm_hidden_state_buf_));
        allocator_->free((void**)(&linear_out_buf_));
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
void XLTransformer2dBlock<T>::forward(std::vector<lyradiff::Tensor>*         output_tensors,
                                      const std::vector<lyradiff::Tensor>*   input_tensors,
                                      const XLTransformer2dBlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}, {"encoder_hidden_states", input_tensors->at(1)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

template<typename T>
void XLTransformer2dBlock<T>::forward(TensorMap*                           output_tensors,
                                      const TensorMap*                     input_tensors,
                                      const XLTransformer2dBlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor output                = output_tensors->at("output");

    size_t batch_size = init_hidden_states.shape[0];
    size_t height     = init_hidden_states.shape[1];
    size_t width      = init_hidden_states.shape[2];

    allocateBuffer(batch_size, height, width);

    // cout << "XLTransformer2dBlock after allocateBuffer" << endl;

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

    // cout << "XLTransformer2dBlock after invokeGroupNorm" << endl;

    // conv1 is replaced by linear in SDXL
    int m_0 = batch_size * height * width;
    int n_0 = inner_dim_;
    int k_0 = inner_dim_;

    // cout << "XLTransformer2dBlock first gemm" << endl;
    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_0,                      // m
                                             m_0,                      // n
                                             k_0,                      // k
                                             weights->proj_in_weight,  // A
                                             k_0,                      // LDA
                                             norm_hidden_state_buf_,   // B
                                             k_0,                      // LDB
                                             linear_out_buf_,          // C
                                             n_0,                      // LDC
                                             weights->proj_in_bias,    // bias
                                                                       //  nullptr,                  // bias
                                             nullptr,                  // residual
                                             1.0f,                     // alpha
                                             0.0f);                    // beta

    TensorMap input_tensor =
        TensorMap(
            {{"hidden_states",
              Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height * width, inner_dim_}, linear_out_buf_)},
             {"encoder_hidden_states", encoder_hidden_states}})
            .setContextThis(input_tensors);

    TensorMap output_tensor({{"output",
                              Tensor(MEMORY_GPU,
                                     init_hidden_states.type,
                                     {batch_size, height * width, inner_dim_},
                                     basic_transformer_res_buf_)}});

    std::vector<T*> bufs = {linear_out_buf_, basic_transformer_res_buf_};

    if (quant_level_ != LyraQuantType::NONE) {
        for (int i = 0; i < inner_trans_num_; i++) {
            // cout << "XLTransformer2dBlock basic_transformer_block " << i << endl;
            transblock_int8_vec[i]->forward(&output_tensor, &input_tensor, weights->transblock_int8_weights[i]);
            // cudaDeviceSynchronize();
            input_tensor.reinit({{"hidden_states",
                                  Tensor(MEMORY_GPU,
                                         init_hidden_states.type,
                                         {batch_size, height * width, inner_dim_},
                                         bufs[(i + 1) % 2])},
                                 {"encoder_hidden_states", encoder_hidden_states}});
            input_tensor.setContextThis(input_tensors);

            output_tensor.reinit(
                {{"output",
                  Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height * width, inner_dim_}, bufs[i % 2])}});
        }
    }
    else {
        for (int i = 0; i < inner_trans_num_; i++) {
            // cout << "XLTransformer2dBlock basic_transformer_block " << i << endl;
            transblock_vec[i]->forward(&output_tensor, &input_tensor, weights->transblock_weights[i]);
            // cudaDeviceSynchronize();
            input_tensor.reinit({{"hidden_states",
                                  Tensor(MEMORY_GPU,
                                         init_hidden_states.type,
                                         {batch_size, height * width, inner_dim_},
                                         bufs[(i + 1) % 2])},
                                 {"encoder_hidden_states", encoder_hidden_states}});
            input_tensor.setContextThis(input_tensors);

            output_tensor.reinit(
                {{"output",
                  Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, height * width, inner_dim_}, bufs[i % 2])}});
        }
    }

    // cout << "XLTransformer2dBlock after all basic transformers" << endl;

    T* output_buf = inner_trans_num_ % 2 == 0 ? linear_out_buf_ : basic_transformer_res_buf_;

    // cout << "XLTransformer2dBlock final gemm" << endl;
    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_0,                             // m
                                             m_0,                             // n
                                             k_0,                             // k
                                             weights->proj_out_weight,        // A
                                             k_0,                             // LDA
                                             output_buf,                      // B
                                             k_0,                             // LDB
                                             output.getPtr<T>(),              // C
                                             n_0,                             // LDC
                                             weights->proj_out_bias,          // bias
                                             init_hidden_states.getPtr<T>(),  // residual
                                             1.0f,                            // alpha
                                             1.0f);                           // beta

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class XLTransformer2dBlock<float>;
template class XLTransformer2dBlock<half>;
}  // namespace lyradiff