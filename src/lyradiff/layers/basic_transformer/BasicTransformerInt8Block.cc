#include "BasicTransformerInt8Block.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/kernels/quant_kernels/ffn_kernels_int8.h"
#include "src/lyradiff/kernels/quant_kernels/layer_norm_int8.h"
#include "src/lyradiff/kernels/quant_kernels/quant_kernels.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/context.h"
#include "src/lyradiff/utils/string_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
BasicTransformerInt8Block<T>::BasicTransformerInt8Block(size_t           dim,
                                                        size_t           num_attention_heads,
                                                        size_t           attention_head_dim,
                                                        size_t           cross_attention_dim,
                                                        cudaStream_t     stream,
                                                        cublasMMWrapper* cublas_wrapper,
                                                        IAllocator*      allocator,
                                                        bool             is_free_buffer_after_forward,
                                                        LyraQuantType    quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false, quant_level),
    dim_(dim),
    num_attention_heads_(num_attention_heads),
    attention_head_dim_(attention_head_dim),
    cross_attention_dim_(cross_attention_dim)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    ffn_inner_dim1_ = dim_ * 8;
    ffn_inner_dim2_ = dim_ * 4;

    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }

    int sm                = getSMVersion();
    cross_attention_layer = new lyradiff::cross_attn::FusedCrossAttentionLayer<T>(num_attention_heads_,
                                                                                attention_head_dim_,
                                                                                sm,
                                                                                getStream(),
                                                                                cublas_wrapper_,
                                                                                allocator_,
                                                                                is_free_buffer_after_forward);

    self_attention_layer = new lyradiff::flash_attn::FusedFlashAttentionLayerV1<T>(num_attention_heads_,
                                                                                 attention_head_dim_,
                                                                                 sm,
                                                                                 getStream(),
                                                                                 cublas_wrapper_,
                                                                                 allocator_,
                                                                                 is_free_buffer_after_forward);

    float      q_scaling      = 1.0;
    const bool force_fp32_acc = false;
    const bool is_s_padded    = false;
    const bool causal_mask    = false;
    const bool has_alibi      = false;
    const bool scal_alibi     = false;
    const int  tp_size        = 1;
    const int  tp_rank        = 0;

    // printf("q_scaling: %f\n", q_scaling);
    // cout << "BasicTransformerBlock created" << endl;

    flash_attn_layer = new lyradiff::flash_attn::FusedFlashAttentionLayerV2<T>(num_attention_heads_,
                                                                             attention_head_dim_,
                                                                             num_attention_heads_,
                                                                             q_scaling,
                                                                             force_fp32_acc,
                                                                             is_s_padded,
                                                                             causal_mask,
                                                                             sm,
                                                                             getStream(),
                                                                             cublas_wrapper_,
                                                                             allocator_,
                                                                             false,
                                                                             has_alibi,
                                                                             scal_alibi,
                                                                             tp_size,
                                                                             tp_rank);

    flash_attn2_layer = new flash_attn2::FlashAttention2Layer<T>(num_attention_heads_,
                                                                 attention_head_dim_,
                                                                 sm,
                                                                 getStream(),
                                                                 cublas_wrapper_,
                                                                 allocator_,
                                                                 is_free_buffer_after_forward);

    int maxM = 16384;
    if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL2
        || this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
        cublas_wrapper_->profileGemmInt8(maxM, dim_ * 3, dim_);
    }
    if (this->quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
        cublas_wrapper_->profileGemmInt8(maxM, dim_, dim_);
    }
    cublas_wrapper_->profileGemmInt8(maxM, dim_ * 8, dim_);
    cublas_wrapper_->profileGemmInt8(maxM, dim_, dim_ * 4);
}

template<typename T>
BasicTransformerInt8Block<T>::BasicTransformerInt8Block(BasicTransformerInt8Block<T> const& basic_transformer_block):
    BaseLayer(basic_transformer_block.stream_,
              basic_transformer_block.cublas_wrapper_,
              basic_transformer_block.allocator_,
              basic_transformer_block.is_free_buffer_after_forward_,
              basic_transformer_block.cuda_device_prop_,
              basic_transformer_block.sparse_),
    dim_(basic_transformer_block.dim_),
    num_attention_heads_(basic_transformer_block.num_attention_heads_),
    attention_head_dim_(basic_transformer_block.attention_head_dim_),
    cross_attention_dim_(basic_transformer_block.cross_attention_dim_),
    ffn_inner_dim1_(basic_transformer_block.ffn_inner_dim1_),
    ffn_inner_dim2_(basic_transformer_block.ffn_inner_dim2_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (std::is_same<T, half>()) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    int sm = getSMVersion();

    cross_attention_layer = basic_transformer_block.cross_attention_layer;
    self_attention_layer  = basic_transformer_block.self_attention_layer;
    flash_attn_layer      = basic_transformer_block.flash_attn_layer;
}

template<typename T>
BasicTransformerInt8Block<T>::~BasicTransformerInt8Block()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete cross_attention_layer;
    delete self_attention_layer;
    delete flash_attn_layer;

    freeBuffer();
}

template<typename T>
void BasicTransformerInt8Block<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(false,
                         "BasicTransformerInt8Block::allocateBuffer() is deprecated. "
                         "Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
size_t BasicTransformerInt8Block<T>::getTotalEncodeSeqLenForAllocBuff(TensorMap* input_map,
                                                                      const BasicTransformerInt8BlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor encoder_hidden_states = input_map->at(ENCODER_HIDDEN_STATES);
    size_t encode_seq_len        = encoder_hidden_states.shape[1];
    return encode_seq_len;
}

template<typename T>
void BasicTransformerInt8Block<T>::allocateBuffer(size_t batch_size,
                                                  size_t seq_len,
                                                  size_t encoder_seq_len,
                                                  size_t ip_encoder_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cur_batch           = batch_size;
    cur_seq_len         = seq_len;
    cur_encoder_seq_len = encoder_seq_len;

    size_t overall_size = 0;

    size_t hidden_state_size      = sizeof(T) * batch_size * seq_len * dim_;
    size_t int8_hidden_state_size = sizeof(int8_t) * batch_size * seq_len * ffn_inner_dim2_;

    size_t self_attn_qkv_size    = sizeof(T) * batch_size * seq_len * 3 * dim_;
    size_t cross_attn_q_size     = sizeof(T) * batch_size * seq_len * dim_;
    size_t cross_attn_kv_size    = sizeof(T) * batch_size * encoder_seq_len * 2 * dim_;
    size_t cross_attn_ip_kv_size = sizeof(T) * batch_size * ip_encoder_seq_len * 2 * dim_;

    size_t ffn_inner_buf1_size = sizeof(T) * batch_size * seq_len * ffn_inner_dim1_;
    size_t ffn_inner_buf2_size = sizeof(T) * batch_size * seq_len * ffn_inner_dim2_;

    // cout << "cur kv size: " << cross_attn_kv_size / 1024.0 / 1024.0  << "MBs " << endl;

    bool use_kv_cache  = getBoolEnvVar("lyradiff_USE_KV_CACHE", false);
    bool is_first_step = getBoolEnvVar("lyradiff_KV_CACHE_FIRST_STEP", false);

    // MACROReMallocWithNameAddOverallSize2(norm_hidden_state_buf_, "BasicTransformerBlock", hidden_state_size, false);
    int8_hidden_state_buf_ = (int8_t*)allocator_->reMallocWithName(
        "BasicTransformerBlock_int8_hidden_state_buf_", int8_hidden_state_size, false);
    MACROReMallocWithNameAddOverallSize2(self_attn_qkv_buf_, "BasicTransformerBlock", self_attn_qkv_size, false);

    if (!use_flash_attn_2) {
        MACROReMallocWithNameAddOverallSize2(self_attn_qkv_buf2_, "BasicTransformerBlock", self_attn_qkv_size, false);
    }

    MACROReMallocWithNameAddOverallSize2(cross_attn_q_buf_, "BasicTransformerBlock", cross_attn_q_size, false);
    MACROReMallocWithNameAddOverallSize2(cross_attn_kv_buf_, "BasicTransformerBlock", cross_attn_kv_size, false);

    if (use_kv_cache) {
        if (is_first_step) {
            cache_cross_attn_kv_buf2_ = (T*)allocator_->reMalloc(cache_cross_attn_kv_buf2_, cross_attn_kv_size, false);
            if (cross_attn_ip_kv_size > 0) {
                cross_attn_ip_kv_buf_ = (T*)allocator_->reMalloc(cross_attn_ip_kv_buf_, cross_attn_ip_kv_size, false);
                is_maintain_ip_buf    = true;
            }
            is_allocate_buffer_ = true;
        }
    }
    else {
        MACROReMallocWithNameAddOverallSize2(
            shared_cross_attn_kv_buf2_, "BasicTransformerBlock", cross_attn_kv_size, false);
    }
    MACROReMallocWithNameAddOverallSize2(attn_output_buf_, "BasicTransformerBlock", hidden_state_size, false);
    MACROReMallocWithNameAddOverallSize2(attn_output_buf2_, "BasicTransformerBlock", hidden_state_size, false);
    MACROReMallocWithNameAddOverallSize2(ffn_inter_buf1_, "BasicTransformerBlock", ffn_inner_buf1_size, false);

    // MACROReMallocWithNameAddOverallSize2(ffn_inter_buf2_, "BasicTransformerBlock", ffn_inner_buf2_size, false);
    // MACROReMallocWithNameAddOverallSize2(ffn_output_buf_, "BasicTransformerBlock", hidden_state_size, false);
}

template<typename T>
void BasicTransformerInt8Block<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)(&cache_cross_attn_kv_buf2_));
        is_allocate_buffer_ = false;
    }

    if (is_maintain_ip_buf) {
        allocator_->free((void**)(&cross_attn_ip_kv_buf_));
    }
}

template<typename T>
void BasicTransformerInt8Block<T>::forward(std::vector<lyradiff::Tensor>*              output_tensors,
                                           const std::vector<lyradiff::Tensor>*        input_tensors,
                                           const BasicTransformerInt8BlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_tensor({{"hidden_states", input_tensors->at(0)}, {"encoder_hidden_states", input_tensors->at(1)}});
    TensorMap output_tensor({{"output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, weights);
}

inline bool existsFile(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

template<typename T>
void BasicTransformerInt8Block<T>::forward(TensorMap*                                output_tensors,
                                           TensorMap*                                input_tensors,
                                           const BasicTransformerInt8BlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor  init_hidden_states    = input_tensors->at("hidden_states");
    Tensor* encoder_hidden_states = &input_tensors->at("encoder_hidden_states");
    Tensor  output                = output_tensors->at("output");

    size_t encoder_seq_total_len = 0;

    if (input_tensors->context_ != nullptr && input_tensors->context_->is_controlnet
        && input_tensors->context_->isValid(PROMPT_IMAGE_EMB)) {
        encoder_hidden_states = &input_tensors->context_->at(PROMPT_IMAGE_EMB);
        encoder_seq_total_len = encoder_hidden_states->shape[1];
    }
    else {
        encoder_seq_total_len = getTotalEncodeSeqLenForAllocBuff(input_tensors, weights);
    }
    // cout << "BasicTransformerBlock after getTotalEncodeSeqLenForAllocBuff" << endl;

    size_t batch_size         = init_hidden_states.shape[0];
    size_t seq_len            = init_hidden_states.shape[1];
    size_t inner_dim          = init_hidden_states.shape[2];
    size_t encoder_seq_len    = encoder_hidden_states->shape[1];
    size_t ip_encoder_seq_len = 0;

    // 判断是否有ipadapter 输入，以及是否load 过 ipadapter 的权重
    bool has_ip = (input_tensors->context_ != nullptr && !input_tensors->context_->is_controlnet
                   && input_tensors->context_->isValid(IP_HIDDEN_STATES) && weights->hasIPAdapter()
                   && input_tensors->context_->getParamVal(IP_RATIO) > 0);

    // cout << "BasicTransformerBlock getTotalEncodeSeqLenForAllocBuff" << endl;

    if (has_ip) {
        Tensor ip_hidden_states = input_tensors->context_->at(IP_HIDDEN_STATES);
        ip_encoder_seq_len      = ip_hidden_states.shape[1];
    }

    allocateBuffer(batch_size, seq_len, encoder_seq_total_len, ip_encoder_seq_len);

    bool use_kv_cache  = getBoolEnvVar("lyradiff_USE_KV_CACHE", false);
    bool is_first_step = getBoolEnvVar("lyradiff_KV_CACHE_FIRST_STEP", false);

    T* cross_attn_kv_buf2_ = shared_cross_attn_kv_buf2_;
    if (use_kv_cache) {
        cross_attn_kv_buf2_ = cache_cross_attn_kv_buf2_;
    }

    // cout << "BasicTransformerBlock allocateBuffer" << endl;

    // T *hidden_states = output.getPtr<T>();
    T* encoder_hidden_states_ptr = encoder_hidden_states->getPtr<T>();

    // cout << "BasicTransformerBlock encoder_hidden_states_ptr" << endl;

    QuantMode cur_quant_mode = cublas_wrapper_->fp16_int8_matmul_runner_->mQuantMode;

    int m = seq_len * batch_size;
    int n = dim_ * 3;
    int k = dim_;

    if (quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL2
        || quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {

        invokeLayerNormInt8O<T>(int8_hidden_state_buf_,
                                init_hidden_states.getPtr<T>(),
                                weights->norm1_gamma,
                                weights->norm1_beta,
                                weights->attention1_qkv_pre_quant_scale,
                                weights->attention1_qkv_input_quant_scale,
                                cur_quant_mode,
                                batch_size,
                                seq_len,
                                dim_,
                                getStream());

        cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                     weights->attention1_qkv_weight_int8,
                                     weights->attention1_qkv_weight_scale,
                                     weights->attention1_qkv_input_quant_scale,
                                     self_attn_qkv_buf_,
                                     m,
                                     n,
                                     k,
                                     (char*)cublas_wrapper_->cublas_workspace_,
                                     stream_);
    }
    else {
        invokeLayerNorm<T>(cross_attn_q_buf_,
                           init_hidden_states.getPtr<T>(),
                           weights->norm1_gamma,
                           weights->norm1_beta,
                           batch_size,
                           seq_len,
                           dim_,
                           getStream());

        int m = seq_len * batch_size;
        int n = dim_ * 3;
        int k = dim_;

        // cout << "BasicTransformerBlock invokeLayerNorm 1" << endl;

        // cout << "m: " << m << " n: " << n << " k: " << k << endl;

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              weights->attention1_qkv_weight,
                              k,
                              cross_attn_q_buf_,
                              k,
                              self_attn_qkv_buf_,
                              n);
    }

    if (!use_flash_attn_2) {
        // 更改 qkv 的shape 从(batch_size, seq_len, 3, head_num, dim_per_head) ->
        // (batch_size, seq_len, head_num, 3, dim_per_head)
        invokeSelfAttnKernelInputPermute<T>(self_attn_qkv_buf2_,
                                            self_attn_qkv_buf_,
                                            batch_size,
                                            seq_len,
                                            dim_,
                                            num_attention_heads_,
                                            attention_head_dim_,
                                            getStream());
        // printf("BasicTransformer after selfAttenInputPermute, do not use flashatten2\n");
        std::vector<Tensor> self_attn_input = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, num_attention_heads_, 3, attention_head_dim_},
                   self_attn_qkv_buf2_)};
        std::vector<Tensor> self_attn_output = {Tensor(MEMORY_GPU,
                                                       init_hidden_states.type,
                                                       {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                                                       attn_output_buf_)};

        // 计算 attention
        // printf("BasicTransformer , do not use flashatten2, call self_atten_layer\n");
        self_attention_layer->forward(&self_attn_output, &self_attn_input);
    }
    else  // 当 seq len > 1024 的时候就可以使用 falsh attn 2
    {
        std::vector<Tensor> self_attn_input = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, 3, num_attention_heads_, attention_head_dim_},
                   self_attn_qkv_buf_)};
        std::vector<Tensor> self_attn_output = {Tensor(MEMORY_GPU,
                                                       init_hidden_states.type,
                                                       {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                                                       attn_output_buf_)};

        // printf("BasicTransformer , use flashatten2, call flash_attn_layer.\n");
        flash_attn_layer->forward(&self_attn_output, &self_attn_input);
    }

    // cout << "BasicTransformerBlock self attn" << endl;

    // self_attn_output_tensor.saveNpy("self_attn_output.npy");

    if (quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
        invokeSqInputQuant(int8_hidden_state_buf_,
                           attn_output_buf_,
                           weights->attention1_to_out_pre_quant_scale,
                           weights->attention1_to_out_input_quant_scale,
                           cur_quant_mode,
                           batch_size,
                           seq_len,
                           dim_,
                           stream_);

        m = seq_len * batch_size;
        n = dim_;
        k = dim_;

        cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                     weights->attention1_to_out_weight_int8,
                                     weights->attention1_to_out_weight_scale,
                                     weights->attention1_to_out_input_quant_scale,
                                     self_attn_qkv_buf_,
                                     m,
                                     n,
                                     k,
                                     (char*)cublas_wrapper_->cublas_workspace_,
                                     stream_);

        invokeFusedResidualBiasLayerNormInt8O(int8_hidden_state_buf_,
                                              output.getPtr<T>(),
                                              self_attn_qkv_buf_,
                                              init_hidden_states.getPtr<T>(),
                                              weights->attention1_to_out_bias,
                                              weights->norm2_gamma,
                                              weights->norm2_beta,
                                              weights->attention2_q_pre_quant_scale,
                                              weights->attention2_q_input_quant_scale,
                                              cur_quant_mode,
                                              batch_size,
                                              seq_len,
                                              dim_,
                                              getStream());

        // 计算 attn2 的 q 和 kv
        m = seq_len * batch_size;
        n = dim_;
        k = dim_;

        cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                     weights->attention2_q_weight_int8,
                                     weights->attention2_q_weight_scale,
                                     weights->attention2_q_input_quant_scale,
                                     cross_attn_q_buf_,
                                     m,
                                     n,
                                     k,
                                     (char*)cublas_wrapper_->cublas_workspace_,
                                     stream_);
    }
    else {
        m = seq_len * batch_size;
        n = dim_;
        k = dim_;

        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n,
                                                 m,
                                                 k,
                                                 weights->attention1_to_out_weight,
                                                 k,
                                                 attn_output_buf_,
                                                 k,
                                                 output.getPtr<T>(),
                                                 n,
                                                 weights->attention1_to_out_bias,  // bias
                                                 init_hidden_states.getPtr<T>(),   // residual
                                                 1.0f,                             // alpha
                                                 1.0f);                            // beta

        // cout << "BasicTransformerBlock to out 1" << endl;
        // output.saveNpy("gt_self_attn_to_out.npy");

        // 计算 norm2
        invokeLayerNorm<T>(self_attn_qkv_buf_,
                           output.getPtr<T>(),
                           weights->norm2_gamma,
                           weights->norm2_beta,
                           batch_size,
                           seq_len,
                           dim_,
                           getStream());
        // cout << "BasicTransformerBlock norm 2" << endl;

        // 计算 attn2 的 q 和 kv
        m = seq_len * batch_size;
        n = dim_;
        k = dim_;

        // cout << "m: " << m << " n: " << n << " k: " << k << endl;

        // cout << "计算 cross attention 的 q" << endl;
        // cout << "m: " << m << " n: " << n << " k: " << k << endl;

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              weights->attention2_q_weight,
                              k,
                              self_attn_qkv_buf_,
                              k,
                              cross_attn_q_buf_,
                              n);
    }

    if ((!use_kv_cache) || (use_kv_cache && is_first_step)) {

        m = encoder_seq_len * batch_size;
        n = dim_ * 2;
        k = cross_attention_dim_;

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              weights->attention2_kv_weight,
                              k,
                              encoder_hidden_states_ptr,
                              k,
                              cross_attn_kv_buf_,
                              n);

        // cout << "BasicTransformerBlock gemm2" << endl;
    
        if (encoder_seq_len <= 128) {
            // cout << "invokeCrossAttnKernelInputPermute" << endl;
            invokeCrossAttnKernelInputPermute<T>(cross_attn_kv_buf2_,
                                                    cross_attn_kv_buf_,
                                                    batch_size,
                                                    encoder_seq_len,
                                                    dim_,
                                                    num_attention_heads_,
                                                    attention_head_dim_,
                                                    getStream());
        }
        else {
            invokeCrossAttn2KernelInputPermute(
                cross_attn_kv_buf2_, cross_attn_kv_buf_, batch_size, encoder_seq_len, dim_, getStream());
        }
        
    }

    // ======================================================================================
    if (encoder_seq_len <= 128) {
        std::vector<Tensor> cross_attn_input = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                   cross_attn_q_buf_),
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, encoder_seq_len, num_attention_heads_, 2, attention_head_dim_},
                   cross_attn_kv_buf2_)};

        std::vector<Tensor> cross_attn_output = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                   attn_output_buf_)};

        cross_attention_layer->forward(&cross_attn_output, &cross_attn_input);
    }
    else {
        // kv_seq 太长。。需要flashattn2
        Tensor q_buf2 = Tensor(MEMORY_GPU,
                               init_hidden_states.type,
                               {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                               cross_attn_q_buf_);
        Tensor k_buf2 = Tensor(MEMORY_GPU,
                               init_hidden_states.type,
                               {batch_size, encoder_seq_len, num_attention_heads_, attention_head_dim_},
                               cross_attn_kv_buf2_);
        Tensor v_buf2 = Tensor(MEMORY_GPU,
                               init_hidden_states.type,
                               {batch_size, encoder_seq_len, num_attention_heads_, attention_head_dim_},
                               &cross_attn_kv_buf2_[k_buf2.size()]);

        Tensor output_tensor = Tensor(MEMORY_GPU,
                                      init_hidden_states.type,
                                      {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                                      attn_output_buf_);

        TensorMap input_map({{"q_buf", q_buf2}, {"k_buf", k_buf2}, {"v_buf", v_buf2}});
        TensorMap output_map({{"attn_output", output_tensor}});
        flash_attn2_layer->forward(&output_map, &input_map);
    }

    // cout << "BasicTransformerBlock attn 2" << endl;

    // kio: reuse memory ipadapter的hidden size=4 [2, 77, 768] [2, 4, 768]
    // 目前ip_adapter scale 默认是1
    // 计算 ipadapter kv_buffer
    // float ip_ratio = input_tensors->context_->getParamVal(IP_RATIO);
    if (has_ip) {
        float  ip_ratio             = input_tensors->context_->getParamVal(IP_RATIO);
        Tensor ip_hidden_states     = input_tensors->context_->at(IP_HIDDEN_STATES);
        T*     ip_hidden_states_ptr = ip_hidden_states.getPtr<T>();

        size_t ip_encoder_seq_len = ip_hidden_states.shape[1];
        m                         = ip_encoder_seq_len * batch_size;

        T* kv_buf_2 = cross_attn_kv_buf2_;
        if (use_kv_cache) {
            kv_buf_2 = cross_attn_ip_kv_buf_;
        }

        if ((!use_kv_cache) || (use_kv_cache && is_first_step)) {

            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  weights->attention2_ip_kv_weight,
                                  k,
                                  ip_hidden_states_ptr,
                                  k,
                                  cross_attn_kv_buf_,
                                  n);

            invokeCrossAttnKernelInputPermute<T>(kv_buf_2,
                                                 cross_attn_kv_buf_,
                                                 batch_size,
                                                 ip_encoder_seq_len,
                                                 dim_,
                                                 num_attention_heads_,
                                                 attention_head_dim_,
                                                 getStream());
        }

        std::vector<Tensor> cross_attn_input = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                   cross_attn_q_buf_),
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, ip_encoder_seq_len, num_attention_heads_, 2, attention_head_dim_},
                   kv_buf_2)};

        std::vector<Tensor> cross_attn_output = {
            Tensor(MEMORY_GPU,
                   init_hidden_states.type,
                   {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                   attn_output_buf2_)};

        cross_attention_layer->forward(&cross_attn_output, &cross_attn_input);

        // ip region mask
        if (input_tensors->context_->isValid(IP_SHALLOW_MASK) && input_tensors->context_->isValid(IP_DEEP_MASK)) {
            T* ip_mask_ptr = (dim_ == 640) ? input_tensors->context_->at(IP_SHALLOW_MASK).getPtr<T>() :
                                             input_tensors->context_->at(IP_DEEP_MASK).getPtr<T>();

            invokeIpMaskAndAddResidual(attn_output_buf_,
                                       attn_output_buf_,
                                       attn_output_buf2_,
                                       ip_mask_ptr,
                                       ip_ratio,
                                       dim_,
                                       seq_len,
                                       batch_size,
                                       getStream());
        }
        else {
            invokeAddResidual(attn_output_buf_,
                              attn_output_buf_,
                              attn_output_buf2_,
                              ip_ratio,
                              dim_,
                              1,
                              seq_len,
                              batch_size,
                              getStream());
        }
    }

    if (quant_level_ == LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3) {
        invokeSqInputQuant(int8_hidden_state_buf_,
                           attn_output_buf_,
                           weights->attention2_to_out_pre_quant_scale,
                           weights->attention2_to_out_input_quant_scale,
                           cur_quant_mode,
                           batch_size,
                           seq_len,
                           dim_,
                           stream_);

        m = seq_len * batch_size;
        n = dim_;
        k = dim_;

        cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                     weights->attention2_to_out_weight_int8,
                                     weights->attention2_to_out_weight_scale,
                                     weights->attention2_to_out_input_quant_scale,
                                     self_attn_qkv_buf_,
                                     m,
                                     n,
                                     k,
                                     (char*)cublas_wrapper_->cublas_workspace_,
                                     stream_);

        invokeFusedResidualBiasLayerNormInt8O(int8_hidden_state_buf_,
                                              output.getPtr<T>(),
                                              self_attn_qkv_buf_,
                                              output.getPtr<T>(),
                                              weights->attention2_to_out_bias,
                                              weights->norm3_gamma,
                                              weights->norm3_beta,
                                              weights->geglu_linear_pre_quant_scale,
                                              weights->geglu_linear_input_quant_scale,
                                              cur_quant_mode,
                                              batch_size,
                                              seq_len,
                                              dim_,
                                              getStream());
    }
    else {
        m = seq_len * batch_size;
        n = dim_;
        k = dim_;
        // 计算 attention2 to_out
        // cout << "计算 attention2 to_out " << endl;
        // cout << "m: " << m << " n: " << n << " k: " << k << endl;

        cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 n,
                                                 m,
                                                 k,
                                                 weights->attention2_to_out_weight,
                                                 k,
                                                 attn_output_buf_,
                                                 k,
                                                 output.getPtr<T>(),
                                                 n,
                                                 weights->attention2_to_out_bias,  // bias
                                                 output.getPtr<T>(),               // residual
                                                 1.0f,                             // alpha
                                                 1.0f);                            // beta

        // 计算 norm3
        // cout << "计算 norm3" << endl;
        invokeLayerNormInt8O<T>(int8_hidden_state_buf_,
                                output.getPtr<T>(),
                                weights->norm3_gamma,
                                weights->norm3_beta,
                                weights->geglu_linear_pre_quant_scale,
                                weights->geglu_linear_input_quant_scale,
                                cur_quant_mode,
                                batch_size,
                                seq_len,
                                dim_,
                                getStream());
    }

    m = seq_len * batch_size;
    n = ffn_inner_dim1_;
    k = dim_;

    cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                 weights->geglu_linear_weight_int8,
                                 weights->geglu_linear_weight_scale,
                                 weights->geglu_linear_input_quant_scale,
                                 ffn_inter_buf1_,
                                 m,
                                 n,
                                 k,
                                 (char*)cublas_wrapper_->cublas_workspace_,
                                 stream_);

    Tensor ffn_inter_buf1 =
        Tensor(MEMORY_GPU, init_hidden_states.type, {batch_size, seq_len, ffn_inner_dim1_}, ffn_inter_buf1_);

    invokeFusedAddBiasGegluInt8O(int8_hidden_state_buf_,
                                 ffn_inter_buf1_,
                                 weights->geglu_linear_bias,
                                 weights->ffn_linear_pre_quant_scale,
                                 weights->ffn_linear_input_quant_scale,
                                 seq_len,
                                 ffn_inner_dim2_,
                                 batch_size,
                                 getStream());

    // 计算 ffn geglu final linear
    m = seq_len * batch_size;
    n = dim_;
    k = ffn_inner_dim2_;
    // cout << "计算 ffn geglu final linear" << endl;
    // cout << "m: " << m << " n: " << n << " k: " << k << endl;
    cublas_wrapper_->runGemmInt8(int8_hidden_state_buf_,
                                 weights->ffn_linear_weight_int8,
                                 weights->ffn_linear_weight_scale,
                                 weights->ffn_linear_input_quant_scale,
                                 ffn_inter_buf1_,
                                 m,
                                 n,
                                 k,
                                 (char*)cublas_wrapper_->cublas_workspace_,
                                 stream_);

    invokeFusedBiasResidualAdd<T>(output.getPtr<T>(),
                                  ffn_inter_buf1_,
                                  output.getPtr<T>(),
                                  weights->ffn_linear_bias,
                                  batch_size,
                                  seq_len,
                                  dim_,
                                  getStream());

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class BasicTransformerInt8Block<float>;
template class BasicTransformerInt8Block<half>;
}  // namespace lyradiff