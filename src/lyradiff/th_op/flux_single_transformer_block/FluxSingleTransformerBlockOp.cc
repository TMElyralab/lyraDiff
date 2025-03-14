#include "FluxSingleTransformerBlockOp.h"

namespace th = torch;
namespace torch_ext {

FluxSingleTransformerBlockOp::FluxSingleTransformerBlockOp(const std::string inference_dtype,
                                                           size_t            embedding_dim,
                                                           size_t            embedding_head_num,
                                                           size_t            embedding_head_dim,
                                                           size_t            mlp_scale,
                                                           const size_t      quant_level):
    inference_dtype_(inference_dtype)
{
    if (inference_dtype_ == "fp32") {
        model_ = new LyraDiffFluxSingleTransformer<float>(
            embedding_dim, embedding_head_num, embedding_head_dim, mlp_scale, quant_level);
    }
    else if (inference_dtype_ == "fp16") {
        model_ = new LyraDiffFluxSingleTransformer<half>(
            embedding_dim, embedding_head_num, embedding_head_dim, mlp_scale, quant_level);
    }
#ifdef ENABLE_BF16
    else if (inference_dtype_ == "bf16") {
        model_ = new LyraDiffFluxSingleTransformer<__nv_bfloat16>(
            embedding_dim, embedding_head_num, embedding_head_dim, mlp_scale, quant_level);
    }
#endif
    else {
        throw "wrong inference_dtype";
    }
}

FluxSingleTransformerBlockOp::~FluxSingleTransformerBlockOp()
{
    delete model_;
}

th::Tensor FluxSingleTransformerBlockOp::forward(th::Tensor hidden_states, th::Tensor rope_emb, th::Tensor temb_emb)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(rope_emb);
    CHECK_TH_CUDA(temb_emb);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(rope_emb);
    CHECK_CONTIGUOUS(temb_emb);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(temb_emb.dtype() == torch::kFloat32, "temb_emb dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(rope_emb.dtype() == torch::kFloat16, "rope_emb dtype should be float16");
        TORCH_CHECK(temb_emb.dtype() == torch::kFloat16, "temb_emb dtype should be float16");
    }
#ifdef ENABLE_BF16
    else if (inference_dtype_ == "bf16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states dtype should be bfloat16");
        TORCH_CHECK(rope_emb.dtype() == torch::kBFloat16, "rope_emb dtype should be bfloat16");
        TORCH_CHECK(temb_emb.dtype() == torch::kBFloat16, "temb_emb dtype should be bfloat16");
    }
#endif
    return model_->forward(hidden_states, rope_emb, temb_emb);
}

void FluxSingleTransformerBlockOp::initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context)
{
    model_->initialize(lyradiff_context);
}

void FluxSingleTransformerBlockOp::reload_model_from_bin(const std::string model_path, const std::string model_dtype)
{
    model_->reload_model_from_bin(model_path, model_dtype);
}

void FluxSingleTransformerBlockOp::reload_model_from_state_dict(const std::string                 prefix,
                                                                th::Dict<std::string, th::Tensor> weights,
                                                                const std::string                 device)
{
    model_->reload_model_from_state_dict(prefix, weights, device);
}

void FluxSingleTransformerBlockOp::clear_lora()
{
    model_->clear_lora();
}

void FluxSingleTransformerBlockOp::load_lora_by_name(std::vector<std::string> lora_name,
                                                     th::Tensor               lora_weights,
                                                     const double             lora_strength)
{
    model_->load_lora_by_name(lora_name, lora_weights, lora_strength);
}

}  // namespace torch_ext
