#include "FluxControlnetModelOp.h"

namespace th = torch;
namespace torch_ext {

FluxControlnetModelOp::FluxControlnetModelOp(const std::string inference_dtype,
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
        model_ = new LyraDiffFluxControlnetModel<float>(input_channels,
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
        model_ = new LyraDiffFluxControlnetModel<half>(input_channels,
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
        model_ = new LyraDiffFluxControlnetModel<__nv_bfloat16>(input_channels,
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

FluxControlnetModelOp::~FluxControlnetModelOp()
{
    delete model_;
}

std::vector<th::Tensor> FluxControlnetModelOp::controlnet_forward(th::Tensor   hidden_states,
                                                                  th::Tensor   encoder_hidden_states,
                                                                  th::Tensor   rope_emb,
                                                                  th::Tensor   pooled_projection,
                                                                  th::Tensor   controlnet_cond,
                                                                  const double timestep,
                                                                  const double guidance,
                                                                  const double controlnet_scale)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_TH_CUDA(rope_emb);
    CHECK_TH_CUDA(pooled_projection);
    CHECK_TH_CUDA(controlnet_cond);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    CHECK_CONTIGUOUS(rope_emb);
    CHECK_CONTIGUOUS(pooled_projection);
    CHECK_CONTIGUOUS(controlnet_cond);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kFloat32, "pooled_projection dtype should be float32");
        TORCH_CHECK(controlnet_cond.dtype() == torch::kFloat32, "controlnet_cond dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kFloat16, "pooled_projection dtype should be float16");
        TORCH_CHECK(controlnet_cond.dtype() == torch::kFloat16, "controlnet_cond dtype should be float16");
    }
#ifdef ENABLE_BF16
    else if (inference_dtype_ == "bf16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states dtype should be bfloat16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kBFloat16,
                    "encoder_hidden_states dtype should be bfloat16");
        TORCH_CHECK(rope_emb.dtype() == torch::kBFloat16, "rope_emb dtype should be bfloat16");
        TORCH_CHECK(pooled_projection.dtype() == torch::kBFloat16, "pooled_projection dtype should be bfloat16");
        TORCH_CHECK(controlnet_cond.dtype() == torch::kBFloat16, "controlnet_cond dtype should be bfloat16");
    }
#endif
    return model_->controlnet_forward(
        hidden_states, encoder_hidden_states, rope_emb, pooled_projection, controlnet_cond, timestep, guidance, controlnet_scale);
}

void FluxControlnetModelOp::initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context)
{
    model_->initialize(lyradiff_context);
}

void FluxControlnetModelOp::reload_controlnet_model(const std::string model_path,
                                                    const int64_t      num_layers,
                                                    const int64_t      num_single_layers,
                                                    const std::string model_dtype)
{
    model_->reload_controlnet_model(model_path, num_layers, num_single_layers, model_dtype);
}

void FluxControlnetModelOp::reload_controlnet_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                                    const int64_t                      num_layers,
                                                                    const int64_t                      num_single_layers,
                                                                    const std::string                 device)
{
    model_->reload_controlnet_model_from_state_dict(weights, num_layers, num_single_layers, device);
}

}  // namespace torch_ext
