#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlock.h"
#include "src/lyradiff/kernels/basic/common_ops.h"
#include "src/lyradiff/kernels/basic_transformer/basic_transformer_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/ffn_kernels.h"
#include "src/lyradiff/kernels/basic_transformer/fused_geglu_kernel.h"
#include "src/lyradiff/kernels/controlnet/residual.h"
#include "src/lyradiff/kernels/norms/layer_norm.h"
#include "src/lyradiff/utils/common_macro_def.h"
#include "src/lyradiff/utils/context.h"
#include "src/lyradiff/utils/string_utils.h"

using namespace std;
namespace lyradiff {

template<typename T>
BasicTransformerBlock<T>::BasicTransformerBlock(size_t           dim,
                                                size_t           num_attention_heads,
                                                size_t           attention_head_dim,
                                                size_t           cross_attention_dim,
                                                cudaStream_t     stream,
                                                cublasMMWrapper* cublas_wrapper,
                                                IAllocator*      allocator,
                                                bool             is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, false),
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
}

template<typename T>
BasicTransformerBlock<T>::BasicTransformerBlock(BasicTransformerBlock<T> const& basic_transformer_block):
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
BasicTransformerBlock<T>::~BasicTransformerBlock()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    delete cross_attention_layer;
    delete self_attention_layer;
    delete flash_attn_layer;

    freeBuffer();
}

template<typename T>
void BasicTransformerBlock<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(false,
                         "BasicTransformerBlock::allocateBuffer() is deprecated. "
                         "Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
size_t BasicTransformerBlock<T>::getTotalEncodeSeqLenForAllocBuff(TensorMap*                            input_map,
                                                                  const BasicTransformerBlockWeight<T>* weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor encoder_hidden_states = input_map->at(ENCODER_HIDDEN_STATES);
    size_t encode_seq_len        = encoder_hidden_states.shape[1];
    return encode_seq_len;
}

template<typename T>
void BasicTransformerBlock<T>::allocateBuffer(size_t batch_size,
                                              size_t seq_len,
                                              size_t encoder_seq_len,
                                              size_t ip_encoder_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cur_batch           = batch_size;
    cur_seq_len         = seq_len;
    cur_encoder_seq_len = encoder_seq_len;

    size_t overall_size = 0;

    size_t hidden_state_size = sizeof(T) * batch_size * seq_len * dim_;

    size_t self_attn_qkv_size    = sizeof(T) * batch_size * seq_len * 3 * dim_;
    size_t cross_attn_q_size     = sizeof(T) * batch_size * seq_len * dim_;
    size_t cross_attn_kv_size    = sizeof(T) * batch_size * encoder_seq_len * 2 * dim_;
    size_t cross_attn_ip_kv_size = sizeof(T) * batch_size * ip_encoder_seq_len * 2 * dim_;

    size_t ffn_inner_buf1_size = sizeof(T) * batch_size * seq_len * ffn_inner_dim1_;
    size_t ffn_inner_buf2_size = sizeof(T) * batch_size * seq_len * ffn_inner_dim2_;

    // cout << "cur kv size: " << cross_attn_kv_size / 1024.0 / 1024.0  << "MBs " << endl;

    bool use_kv_cache  = getBoolEnvVar("LYRADIFF_USE_KV_CACHE", false);
    bool is_first_step = getBoolEnvVar("LYRADIFF_KV_CACHE_FIRST_STEP", false);

    MACROReMallocWithNameAddOverallSize2(norm_hidden_state_buf_, "BasicTransformerBlock", hidden_state_size, false);
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
    if (!use_fused_geglu) {
        MACROReMallocWithNameAddOverallSize2(ffn_inter_buf1_, "BasicTransformerBlock", ffn_inner_buf1_size, false);
    }
    MACROReMallocWithNameAddOverallSize2(ffn_inter_buf2_, "BasicTransformerBlock", ffn_inner_buf2_size, false);
}

template<typename T>
void BasicTransformerBlock<T>::freeBuffer()
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
void BasicTransformerBlock<T>::forward(std::vector<lyradiff::Tensor>*        output_tensors,
                                       const std::vector<lyradiff::Tensor>*  input_tensors,
                                       const BasicTransformerBlockWeight<T>* weights)
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
void BasicTransformerBlock<T>::forward(TensorMap*                            output_tensors,
                                       TensorMap*                            input_tensors,
                                       const BasicTransformerBlockWeight<T>* weights)
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
    size_t batch_size         = init_hidden_states.shape[0];
    size_t seq_len            = init_hidden_states.shape[1];
    size_t inner_dim          = init_hidden_states.shape[2];
    size_t encoder_seq_len    = encoder_hidden_states->shape[1];
    size_t ip_encoder_seq_len = 0;

    // check if has ipadapter inputs, and has loaded ipadapter weight
    bool has_ip = (input_tensors->context_ != nullptr && !input_tensors->context_->is_controlnet
                   && input_tensors->context_->isValid(IP_HIDDEN_STATES) && weights->hasIPAdapter()
                   && input_tensors->context_->getParamVal(IP_RATIO) > 0);

    if (has_ip) {
        Tensor ip_hidden_states = input_tensors->context_->at(IP_HIDDEN_STATES);
        ip_encoder_seq_len      = ip_hidden_states.shape[1];
    }
    allocateBuffer(batch_size, seq_len, encoder_seq_total_len, ip_encoder_seq_len);
    bool use_kv_cache  = getBoolEnvVar("LYRADIFF_USE_KV_CACHE", false);
    bool is_first_step = getBoolEnvVar("LYRADIFF_KV_CACHE_FIRST_STEP", false);

    T* cross_attn_kv_buf2_ = shared_cross_attn_kv_buf2_;
    if (use_kv_cache) {
        cross_attn_kv_buf2_ = cache_cross_attn_kv_buf2_;
    }

    // T *hidden_states = output.getPtr<T>();
    T* encoder_hidden_states_ptr = encoder_hidden_states->getPtr<T>();

    // 计算norm1
    invokeLayerNorm<T>(norm_hidden_state_buf_,
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

    T* qkv_inp_buf = norm_hidden_state_buf_;

    cublas_wrapper_->Gemm(
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, weights->attention1_qkv_weight, k, qkv_inp_buf, k, self_attn_qkv_buf_, n);

    Tensor self_attn_qkv = Tensor(MEMORY_GPU,
                                  init_hidden_states.type,
                                  {batch_size, seq_len, 3, num_attention_heads_, attention_head_dim_},
                                  self_attn_qkv_buf_);

    // self_attn_qkv.saveNpy("gt_qkv_res.npy");
    // printf("attn2: %d %d\n", num_attention_heads_, attention_head_dim_);

    if (!use_flash_attn_2) {
        // updaye qkv shape from (batch_size, seq_len, 3, head_num, dim_per_head) ->
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

        // calc attention
        // printf("BasicTransformer , do not use flashatten2, call self_atten_layer\n");
        self_attention_layer->forward(&self_attn_output, &self_attn_input);
    }
    else  // when seq len > 1024 use falsh attn 2
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

    invokeLayerNorm<T>(norm_hidden_state_buf_,
                       output.getPtr<T>(),
                       weights->norm2_gamma,
                       weights->norm2_beta,
                       batch_size,
                       seq_len,
                       dim_,
                       getStream());

    m = seq_len * batch_size;
    n = dim_;
    k = dim_;

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          weights->attention2_q_weight,
                          k,
                          norm_hidden_state_buf_,
                          k,
                          cross_attn_q_buf_,
                          n);

    // cout << "m: " << m << " n: " << n << " k: " << k << endl;
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

    Tensor attn2_output = Tensor(MEMORY_GPU,
                                 init_hidden_states.type,
                                 {batch_size, seq_len, num_attention_heads_, attention_head_dim_},
                                 attn_output_buf_);

    m = seq_len * batch_size;
    n = dim_;
    k = dim_;

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

    T* norm3_inp = output.getPtr<T>();

    invokeLayerNorm<T>(norm_hidden_state_buf_,
                       norm3_inp,
                       weights->norm3_gamma,
                       weights->norm3_beta,
                       batch_size,
                       seq_len,
                       dim_,
                       getStream());

    T* basic_ln3_buf = norm_hidden_state_buf_;
    // ffn geglu
    bool has_context      = input_tensors->context_ != nullptr;
    bool map_de_mods_size = 0;
    if (has_context) {
        auto lora_container =
            weight_loader_manager_glob->map_lora_container[input_tensors->context_->cur_running_module];
        map_de_mods_size = lora_container.map_de_mods.size();
    }

    if (map_de_mods_size == 0) {
        fused_linear_geglu(
            ffn_inter_buf2_,
            basic_ln3_buf,
            weights->geglu_linear_weight,                           // left side gemm weight
            weights->geglu_linear_bias,                             // left side gemm bias
            &weights->geglu_linear_weight[dim_ * ffn_inner_dim2_],  // right side gemm weight
            &weights->geglu_linear_bias[ffn_inner_dim2_],           //  right side gemm bias
            batch_size,                                             // input batchsize
            seq_len,                                                // input seqlen
            dim_,                                                   // input dim
            ffn_inner_dim2_,                                        // output dim
            cublas_wrapper_->cublas_workspace_,                     // workspace
            false,  // whether use low precision(fp16) as accumulater, will increase speed but loss accuracy
            getStream());
    }
    else {
        // 计算 ffn geglu
        m = seq_len * batch_size;
        n = ffn_inner_dim1_;
        k = dim_;
        // cout << "计算 ffn geglu" << endl;
        // cout << "m: " << m << " n: " << n << " k: " << k << endl;

        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              weights->geglu_linear_weight,
                              k,
                              norm_hidden_state_buf_,
                              k,
                              ffn_inter_buf1_,
                              n);
        // 跑 geglu + bias
        // cout << "计算 geglu + bias " << endl;
        invokeFusedAddBiasGeglu(ffn_inter_buf2_,
                                ffn_inter_buf1_,
                                weights->geglu_linear_bias,
                                seq_len,
                                ffn_inner_dim2_,
                                batch_size,
                                getStream());
    }

    // 计算 ffn geglu final linear
    m = seq_len * batch_size;
    n = dim_;
    k = ffn_inner_dim2_;
    // cout << "计算 ffn geglu final linear" << endl;

    // cout << "m: " << m << " n: " << n << " k: " << k << endl;
    cublas_wrapper_->GemmWithResidualAndBias(CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             n,
                                             m,
                                             k,
                                             weights->ffn_linear_weight,
                                             k,
                                             ffn_inter_buf2_,
                                             k,
                                             output.getPtr<T>(),
                                             n,
                                             weights->ffn_linear_bias,  // bias
                                             norm3_inp,                 // residual
                                             1.0f,                      // alpha
                                             1.0f);                     // beta

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class BasicTransformerBlock<float>;
template class BasicTransformerBlock<half>;
}  // namespace lyradiff