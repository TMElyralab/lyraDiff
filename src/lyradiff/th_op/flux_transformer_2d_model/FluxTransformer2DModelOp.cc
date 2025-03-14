#include "FluxTransformer2DModelOp.h"

namespace th = torch;
namespace torch_ext {

FluxTransformer2DModelOp::FluxTransformer2DModelOp(const std::string inference_dtype,
                                                   const size_t      input_channels,
                                                   const size_t      num_layers,
                                                   const size_t      num_single_layers,
                                                   const size_t      attention_head_dim,
                                                   const size_t      num_attention_heads,
                                                   const size_t      pooled_projection_dim,
                                                   const size_t      joint_attention_dim,
                                                   const bool        guidance_embeds,
                                                   const size_t      quant_level):
    inference_dtype_(inference_dtype)
{

    if (inference_dtype_ == "fp32") {
        model_ = new LyraDiffFluxTransformer2DModel<float>(input_channels,
                                                         num_layers,
                                                         num_single_layers,
                                                         attention_head_dim,
                                                         num_attention_heads,
                                                         pooled_projection_dim,
                                                         joint_attention_dim,
                                                         guidance_embeds,
                                                         quant_level);
    }
    else if (inference_dtype_ == "fp16") {
        model_ = new LyraDiffFluxTransformer2DModel<half>(input_channels,
                                                        num_layers,
                                                        num_single_layers,
                                                        attention_head_dim,
                                                        num_attention_heads,
                                                        pooled_projection_dim,
                                                        joint_attention_dim,
                                                        guidance_embeds,
                                                        quant_level);
    }
#ifdef ENABLE_BF16
    else if (inference_dtype_ == "bf16") {
        model_ = new LyraDiffFluxTransformer2DModel<__nv_bfloat16>(input_channels,
                                                                 num_layers,
                                                                 num_single_layers,
                                                                 attention_head_dim,
                                                                 num_attention_heads,
                                                                 pooled_projection_dim,
                                                                 joint_attention_dim,
                                                                 guidance_embeds,
                                                                 quant_level);
    }
#endif
    else {
        throw "wrong inference_dtype";
    }
}

FluxTransformer2DModelOp::~FluxTransformer2DModelOp()
{
    delete model_;
}

th::Tensor
FluxTransformer2DModelOp::transformer_forward(th::Tensor                       hidden_states,
                                              th::Tensor                       encoder_hidden_states,
                                              th::Tensor                       rope_emb,
                                              th::Tensor                       pooled_projection,
                                              const double                     timestep,
                                              const double                     guidance,
                                              th::optional<vector<th::Tensor>> controlnet_block_samples,
                                              th::optional<vector<th::Tensor>> controlnet_single_block_samples,
                                              th::optional<bool>               controlnet_blocks_repeat)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_TH_CUDA(rope_emb);
    CHECK_TH_CUDA(pooled_projection);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    CHECK_CONTIGUOUS(rope_emb);
    CHECK_CONTIGUOUS(pooled_projection);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kFloat32, "pooled_projection dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kFloat16, "pooled_projection dtype should be float16");
    }
#ifdef ENABLE_BF16
    else if (inference_dtype_ == "bf16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states dtype should be bfloat16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kBFloat16,
                    "encoder_hidden_states dtype should be bfloat16");
        TORCH_CHECK(rope_emb.dtype() == torch::kBFloat16, "rope_emb dtype should be bfloat16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kBFloat16, "pooled_projection dtype should be bfloat16");
    }
#endif
    return model_->transformer_forward(hidden_states,
                                       encoder_hidden_states,
                                       rope_emb,
                                       pooled_projection,
                                       timestep,
                                       guidance,
                                       controlnet_block_samples,
                                       controlnet_single_block_samples,
                                       controlnet_blocks_repeat);
}

// void XLUnet2dConditionalModelOp::load_lora(const std::string lora_path,
//                                            const std::string lora_name,
//                                            const double      lora_strength,
//                                            const std::string model_dtype)
// {
//     model_->load_lora(lora_path, lora_name, lora_strength, model_dtype);
// }

// void XLUnet2dConditionalModelOp::unload_lora(const std::string lora_name, const bool clean_mem_cache)
// {
//     model_->unload_lora(lora_name, clean_mem_cache);
// }

void FluxTransformer2DModelOp::initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context)
{
    model_->initialize(lyradiff_context);
}

void FluxTransformer2DModelOp::reload_transformer_model(const std::string model_path, const std::string model_dtype)
{
    model_->reload_transformer_model(model_path, model_dtype);
}

void FluxTransformer2DModelOp::reload_transformer_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                                        const std::string                 device)
{
    model_->reload_transformer_model_from_state_dict(weights, device);
}

void FluxTransformer2DModelOp::clear_lora()
{
    model_->clear_lora();
}

void FluxTransformer2DModelOp::load_lora_by_name(std::vector<std::string> lora_name,
                                                 th::Tensor               lora_weights,
                                                 const double             lora_strength)
{
    model_->load_lora_by_name(lora_name, lora_weights, lora_strength);
}

}  // namespace torch_ext
