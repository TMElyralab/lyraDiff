#include "FluxTransformer2DModel.h"
// #include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/flux_transformer_block/flux_transformer_kernels.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/string_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
FluxTransformer2DModel<T>::FluxTransformer2DModel(cudaStream_t        stream,
                                                  cublasMMWrapper*    cublas_wrapper,
                                                  IAllocator*         allocator,
                                                  const bool          is_free_buffer_after_forward,
                                                  const bool          sparse,
                                                  const size_t        input_channels,
                                                  const size_t        num_layers,
                                                  const size_t        num_single_layers,
                                                  const size_t        attention_head_dim,
                                                  const size_t        num_attention_heads,
                                                  const size_t        pooled_projection_dim,
                                                  const size_t        joint_attention_dim,
                                                  const bool          guidance_embeds,
                                                  const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    input_channels_(input_channels),
    num_layers_(num_layers),
    num_single_layers_(num_single_layers),
    attention_head_dim_(attention_head_dim),
    num_attention_heads_(num_attention_heads),
    pooled_projection_dim_(pooled_projection_dim),
    joint_attention_dim_(joint_attention_dim),
    guidance_embeds_(guidance_embeds)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }

    embedding_dim_ = attention_head_dim_ * num_attention_heads_;

    timestep_embedding = new CombinedTimestepGuidanceTextProjEmbeddings<T>(pooled_projection_dim_,
                                                                           embedding_dim_,
                                                                           embedding_input_dim_,
                                                                           true,
                                                                           stream,
                                                                           cublas_wrapper,
                                                                           allocator,
                                                                           is_free_buffer_after_forward,
                                                                           false);

    norm_out = new AdaLayerNorm<T>(
        embedding_dim_, 2, true, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, false);

    if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL || this->quant_level_ == LyraQuantType::FP8_W8A8) {
        transformer_block = new FluxTransformerFP8Block<T>(embedding_dim_,
                                                           num_attention_heads_,
                                                           attention_head_dim_,
                                                           -1,
                                                           stream,
                                                           cublas_wrapper,
                                                           allocator,
                                                           is_free_buffer_after_forward,
                                                           false,
                                                           quant_level_);

        single_transformer_block = new FluxSingleTransformerFP8Block<T>(embedding_dim_,
                                                                        num_attention_heads_,
                                                                        attention_head_dim_,
                                                                        4,
                                                                        stream,
                                                                        cublas_wrapper,
                                                                        allocator,
                                                                        is_free_buffer_after_forward,
                                                                        false,
                                                                        quant_level_);
    }
    else if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL || this->quant_level_ == LyraQuantType::INT4_W4A4) {
        transformer_block = new FluxTransformerInt4Block<T>(embedding_dim_,
                                                            num_attention_heads_,
                                                            attention_head_dim_,
                                                            -1,
                                                            stream,
                                                            cublas_wrapper,
                                                            allocator,
                                                            is_free_buffer_after_forward,
                                                            false,
                                                            quant_level_);

        single_transformer_block = new FluxSingleTransformerInt4Block<T>(embedding_dim_,
                                                                         num_attention_heads_,
                                                                         attention_head_dim_,
                                                                         4,
                                                                         stream,
                                                                         cublas_wrapper,
                                                                         allocator,
                                                                         is_free_buffer_after_forward,
                                                                         false,
                                                                         quant_level_);
    }
    else {
        transformer_block = new FluxTransformerBlock<T>(embedding_dim_,
                                                        num_attention_heads_,
                                                        attention_head_dim_,
                                                        -1,
                                                        stream,
                                                        cublas_wrapper,
                                                        allocator,
                                                        is_free_buffer_after_forward,
                                                        false,
                                                        quant_level_);

        single_transformer_block = new FluxSingleTransformerBlock<T>(embedding_dim_,
                                                                     num_attention_heads_,
                                                                     attention_head_dim_,
                                                                     4,
                                                                     stream,
                                                                     cublas_wrapper,
                                                                     allocator,
                                                                     is_free_buffer_after_forward,
                                                                     false,
                                                                     quant_level_);
    }
}

template<typename T>
FluxTransformer2DModel<T>::FluxTransformer2DModel(FluxTransformer2DModel<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_),
    input_channels_(other.input_channels_),
    num_layers_(other.num_layers_),
    num_single_layers_(other.num_single_layers_),
    attention_head_dim_(other.attention_head_dim_),
    num_attention_heads_(other.num_attention_heads_),
    pooled_projection_dim_(other.pooled_projection_dim_),
    joint_attention_dim_(other.joint_attention_dim_),
    guidance_embeds_(other.guidance_embeds_)
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
    timestep_embedding       = other.timestep_embedding;
    norm_out                 = other.norm_out;
    transformer_block        = other.transformer_block;
    single_transformer_block = other.single_transformer_block;
}

template<typename T>
void FluxTransformer2DModel<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false,
        "FluxTransformer2DModel::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void FluxTransformer2DModel<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t encoder_seq_len)
{
    size_t temb_size           = sizeof(T) * batch_size * embedding_dim_;
    size_t hidden_size         = sizeof(T) * batch_size * seq_len * embedding_dim_;
    size_t encoder_hidden_size = sizeof(T) * batch_size * encoder_seq_len * embedding_dim_;
    size_t fused_hidden_size   = sizeof(T) * batch_size * (seq_len + encoder_seq_len) * embedding_dim_;
    size_t msa_size            = sizeof(T) * batch_size * 2 * embedding_dim_;

    temb_buf_        = (T*)allocator_->reMalloc(temb_buf_, temb_size, false);
    hidden_buf_1     = (T*)allocator_->reMalloc(hidden_buf_1, hidden_size, false);
    hidden_buf_2     = (T*)allocator_->reMalloc(hidden_buf_2, hidden_size, false);
    encoder_buf_1    = (T*)allocator_->reMalloc(encoder_buf_1, encoder_hidden_size, false);
    encoder_buf_2    = (T*)allocator_->reMalloc(encoder_buf_2, encoder_hidden_size, false);
    cat_hidden_buf_1 = (T*)allocator_->reMalloc(cat_hidden_buf_1, fused_hidden_size, false);
    cat_hidden_buf_2 = (T*)allocator_->reMalloc(cat_hidden_buf_2, fused_hidden_size, false);

    encoder_buf_ = (T*)allocator_->reMalloc(encoder_buf_, encoder_hidden_size, false);
    msa_buf_     = (T*)allocator_->reMalloc(msa_buf_, msa_size, false);
}

template<typename T>
void FluxTransformer2DModel<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&temb_buf_));
        allocator_->free((void**)(&hidden_buf_1));
        allocator_->free((void**)(&hidden_buf_2));
        allocator_->free((void**)(&encoder_buf_1));
        allocator_->free((void**)(&encoder_buf_2));

        allocator_->free((void**)(&msa_buf_));
        allocator_->free((void**)(&cat_hidden_buf_1));
        allocator_->free((void**)(&cat_hidden_buf_2));

        temb_buf_     = nullptr;
        hidden_buf_1  = nullptr;
        hidden_buf_2  = nullptr;
        encoder_buf_1 = nullptr;
        encoder_buf_2 = nullptr;

        msa_buf_         = nullptr;
        cat_hidden_buf_1 = nullptr;
        cat_hidden_buf_2 = nullptr;

        // allocator_->freeAllNameBuf();
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void FluxTransformer2DModel<T>::transformer_forward(TensorMap*                             output_tensors,
                                                    const TensorMap*                       input_tensors,
                                                    const float                            timestep,
                                                    const float                            guidance,
                                                    const FluxTransformer2DModelWeight<T>* weights,
                                                    const std::vector<Tensor>&             controlnet_block_samples,
                                                    const std::vector<Tensor>& controlnet_single_block_samples,
                                                    const bool                 controlnet_blocks_repeat)
{
    Tensor init_hidden_states    = input_tensors->at("hidden_states");
    Tensor encoder_hidden_states = input_tensors->at("encoder_hidden_states");
    Tensor rope_embs             = input_tensors->at("rope_embs");
    Tensor pooled_projection     = input_tensors->at("pooled_projection");
    Tensor output                = output_tensors->at("output");

    size_t batch_size      = init_hidden_states.shape[0];
    size_t seq_len         = init_hidden_states.shape[1];
    size_t encoder_seq_len = encoder_hidden_states.shape[1];

    // 如果 height 和 width 一致，这里不需要再次 allocate

    LYRA_CHECK_WITH_INFO(controlnet_single_block_samples.size() == 0,
                         "controlnet_single_block_samples currently needs to be 0");
    bool   has_controlnet   = controlnet_block_samples.size() > 0;
    size_t interval_control = 0;
    if (has_controlnet) {
        interval_control = (num_layers_ - 1) / controlnet_block_samples.size() + 1;
    }
    allocateBuffer(batch_size, seq_len, encoder_seq_len);

    Tensor temb_tensor = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, embedding_dim_}, temb_buf_);

    TensorMap input_tensor  = TensorMap({{"pooled_projection", pooled_projection}});
    TensorMap output_tensor = TensorMap({{"output", temb_tensor}});

    // cout << "timestep: " << timestep << endl;
    // cout << "guidance: " << guidance << endl;

    timestep_embedding->forward(&output_tensor, &input_tensor, timestep, guidance, weights->timestep_embedding_weight);

    // temb_tensor.saveNpy("temb_tensor.npy");

    // cout << "timestep_embedding" << endl;
    int m_1 = batch_size * seq_len;
    int n_1 = embedding_dim_;
    int k_1 = input_channels_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                             // m
                                             m_1,                             // n
                                             k_1,                             // k
                                             weights->x_embedder_weight,      // A
                                             k_1,                             // LDA
                                             init_hidden_states.getPtr<T>(),  // B
                                             k_1,                             // LDB
                                             hidden_buf_1,                    // C
                                             n_1,                             // LDC
                                             weights->x_embedder_bias,        // bias
                                             nullptr,                         // residual
                                             1.0f,                            // alpha
                                             0.0f);                           // beta

    Tensor h_tensor = Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_buf_1);
    // h_tensor.saveNpy("x_embedding.npy");
    // bool is_first_step = getBoolEnvVar("lyradiff_KV_CACHE_FIRST_STEP", false);

    // if (is_first_step) {
    m_1 = batch_size * encoder_seq_len;
    n_1 = embedding_dim_;
    k_1 = joint_attention_dim_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                                // m
                                             m_1,                                // n
                                             k_1,                                // k
                                             weights->context_embedder_weight,   // A
                                             k_1,                                // LDA
                                             encoder_hidden_states.getPtr<T>(),  // B
                                             k_1,                                // LDB
                                             encoder_buf_1,                      // C
                                             n_1,                                // LDC
                                             weights->context_embedder_bias,     // bias
                                             nullptr,                            // residual
                                             1.0f,                               // alpha
                                             0.0f);                              // beta

    Tensor c_tensor =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, encoder_seq_len, embedding_dim_}, encoder_buf_1);
    // c_tensor.saveNpy("context_embedding.npy");

    T* hidden_input_ptr   = hidden_buf_1;
    T* encoder_input_ptr  = encoder_buf_1;
    T* hidden_output_ptr  = hidden_buf_2;
    T* encoder_output_ptr = encoder_buf_2;

    for (int i = 0; i < num_layers_; i++) {
        // 判断当前输入buffer是啥

        if (i % 2 == 1) {
            hidden_input_ptr   = hidden_buf_2;
            encoder_input_ptr  = encoder_buf_2;
            hidden_output_ptr  = hidden_buf_1;
            encoder_output_ptr = encoder_buf_1;
        }
        // else if (i == 0) {
        //     hidden_input_ptr   = hidden_buf_1;
        //     encoder_input_ptr  = encoder_buf_;
        //     hidden_output_ptr  = hidden_buf_2;
        //     encoder_output_ptr = encoder_buf_2;
        // }
        else {
            hidden_input_ptr   = hidden_buf_1;
            encoder_input_ptr  = encoder_buf_1;
            hidden_output_ptr  = hidden_buf_2;
            encoder_output_ptr = encoder_buf_2;
        }

        Tensor hidden_input =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_input_ptr);
        Tensor encoder_input = Tensor(
            MEMORY_GPU, init_hidden_states.type, {batch_size, encoder_seq_len, embedding_dim_}, encoder_input_ptr);

        Tensor hidden_output =
            Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_output_ptr);
        Tensor encoder_output = Tensor(
            MEMORY_GPU, init_hidden_states.type, {batch_size, encoder_seq_len, embedding_dim_}, encoder_output_ptr);

        input_tensor = TensorMap({{"input", hidden_input},
                                  {"encoder_input", encoder_input},
                                  {"temb", temb_tensor},
                                  {"rope_emb", rope_embs}});

        output_tensor = TensorMap({{"output", hidden_output}, {"encoder_output", encoder_output}});

        if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL || this->quant_level_ == LyraQuantType::FP8_W8A8) {
            FluxTransformerFP8Block<T>* block = (FluxTransformerFP8Block<T>*)transformer_block;
            block->forward(
                &output_tensor, &input_tensor, (FluxTransformerFP8BlockWeight<T>*)weights->transformer_block_weight[i]);
        }
        else if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL
                 || this->quant_level_ == LyraQuantType::INT4_W4A4) {
            FluxTransformerInt4Block<T>* block = (FluxTransformerInt4Block<T>*)transformer_block;
            block->forward(&output_tensor,
                           &input_tensor,
                           (FluxTransformerInt4BlockWeight<T>*)weights->transformer_block_weight[i]);
        }
        else {
            transformer_block->forward(&output_tensor, &input_tensor, weights->transformer_block_weight[i]);
        }

        if (has_controlnet) {
            T* cur_controlnet_ptr = controlnet_block_samples[i / interval_control].getPtr<T>();
            if (controlnet_blocks_repeat) {
                cur_controlnet_ptr = controlnet_block_samples[i % controlnet_block_samples.size()].getPtr<T>();
            }
            invokeLoadLora(hidden_output_ptr, cur_controlnet_ptr, hidden_output.size(), 1.0, stream_);
        }
    }
    // cout << "transformer_block" << endl;

    Tensor transformer_hidden_output =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_output_ptr);
    Tensor transformer_encoder_output =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, encoder_seq_len, embedding_dim_}, encoder_output_ptr);

    // transformer_hidden_output.saveNpy("transformer_hidden_output.npy");
    // transformer_encoder_output.saveNpy("transformer_encoder_output.npy");

    invokeCatEncoderAndHidden(cat_hidden_buf_1,
                              encoder_output_ptr,
                              hidden_output_ptr,
                              batch_size,
                              seq_len,
                              encoder_seq_len,
                              embedding_dim_,
                              stream_);

    Tensor cat_hidden = Tensor(
        MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len + encoder_seq_len, embedding_dim_}, cat_hidden_buf_1);

    // cat_hidden.saveNpy("cat_hidden.npy");

    hidden_input_ptr  = cat_hidden_buf_1;
    hidden_output_ptr = cat_hidden_buf_2;

    for (int i = 0; i < num_single_layers_; i++) {
        // 判断当前输入buffer是啥

        if (i % 2 == 1) {
            hidden_input_ptr  = cat_hidden_buf_2;
            hidden_output_ptr = cat_hidden_buf_1;
        }
        else {
            hidden_input_ptr  = cat_hidden_buf_1;
            hidden_output_ptr = cat_hidden_buf_2;
        }

        Tensor hidden_input = Tensor(MEMORY_GPU,
                                     init_hidden_states.type,
                                     {batch_size, seq_len + encoder_seq_len, embedding_dim_},
                                     hidden_input_ptr);

        Tensor hidden_output = Tensor(MEMORY_GPU,
                                      init_hidden_states.type,
                                      {batch_size, seq_len + encoder_seq_len, embedding_dim_},
                                      hidden_output_ptr);

        input_tensor  = TensorMap({{"input", hidden_input}, {"temb", temb_tensor}, {"rope_emb", rope_embs}});
        output_tensor = TensorMap({{"output", hidden_output}});

        if (this->quant_level_ == LyraQuantType::FP8_W8A8_FULL || this->quant_level_ == LyraQuantType::FP8_W8A8) {
            FluxSingleTransformerFP8Block<T>* block = (FluxSingleTransformerFP8Block<T>*)single_transformer_block;
            block->forward(&output_tensor,
                           &input_tensor,
                           (FluxSingleTransformerFP8BlockWeight<T>*)weights->single_transformer_block_weight[i]);
        }
        else if (this->quant_level_ == LyraQuantType::INT4_W4A4_FULL
                 || this->quant_level_ == LyraQuantType::INT4_W4A4) {
            FluxSingleTransformerInt4Block<T>* block = (FluxSingleTransformerInt4Block<T>*)single_transformer_block;
            block->forward(&output_tensor,
                           &input_tensor,
                           (FluxSingleTransformerInt4BlockWeight<T>*)weights->single_transformer_block_weight[i]);
        }
        else {
            single_transformer_block->forward(
                &output_tensor, &input_tensor, weights->single_transformer_block_weight[i]);
        }
    }
    // cout << "single_transformer_block" << endl;

    Tensor single_output = Tensor(MEMORY_GPU,
                                  init_hidden_states.type,
                                  {batch_size, seq_len + encoder_seq_len, embedding_dim_},
                                  hidden_output_ptr);
    // single_output.saveNpy("single_output.npy");

    invokeSpiltEncoderAndHidden(
        hidden_buf_1, encoder_buf_1, hidden_output_ptr, batch_size, seq_len, encoder_seq_len, embedding_dim_, stream_);

    Tensor hidden_input =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_buf_1);

    // hidden_input.saveNpy("splited_hidden.npy");

    Tensor msa_output = Tensor(MEMORY_GPU, init_hidden_states.type, {2, batch_size, embedding_dim_}, msa_buf_);

    Tensor hidden_output =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, embedding_dim_}, hidden_buf_2);

    input_tensor  = TensorMap({{"input", hidden_input}, {"temb", temb_tensor}});
    output_tensor = TensorMap({{"output", hidden_output}, {"msa_output", msa_output}});

    norm_out->forward(&output_tensor, &input_tensor, weights->norm_weight);
    // cout << "norm_out" << endl;

    // hidden_output.saveNpy("norm_out.npy");

    m_1 = batch_size * seq_len;
    n_1 = input_channels_;
    k_1 = embedding_dim_;

    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n_1,                       // m
                                             m_1,                       // n
                                             k_1,                       // k
                                             weights->proj_out_weight,  // A
                                             k_1,                       // LDA
                                             hidden_buf_2,              // B
                                             k_1,                       // LDB
                                             output.getPtr<T>(),        // C
                                             n_1,                       // LDC
                                             weights->proj_out_bias,    // bias
                                             nullptr,                   // residual
                                             1.0f,                      // alpha
                                             0.0f);                     // beta

    // allocator_->printAllNameSize();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
FluxTransformer2DModel<T>::~FluxTransformer2DModel()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;

    delete timestep_embedding;
    delete norm_out;
    delete single_transformer_block;
    delete transformer_block;

    timestep_embedding       = nullptr;
    norm_out                 = nullptr;
    single_transformer_block = nullptr;
    transformer_block        = nullptr;

    freeBuffer();
}

template class FluxTransformer2DModel<float>;
template class FluxTransformer2DModel<half>;
#ifdef ENABLE_BF16
template class FluxTransformer2DModel<__nv_bfloat16>;
#endif
}  // namespace lyradiff