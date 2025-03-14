#include "GLVUnet2dConditionalModelOp.h"

namespace th = torch;
namespace torch_ext {

GLVUnet2dConditionalModelOp::GLVUnet2dConditionalModelOp(const std::string inference_dtype,
                                                         const int64_t     input_channels,
                                                         const int64_t     output_channels):
    inference_dtype_(inference_dtype)
{

    if (inference_dtype_ == "fp32") {
        model_ = new lyradiffGLVUnet2dConditionalModel<float>(inference_dtype, input_channels, output_channels);
    }
    else if (inference_dtype_ == "fp16") {
        model_ = new lyradiffGLVUnet2dConditionalModel<half>(inference_dtype, input_channels, output_channels);
    }
    else {
        throw "wrong inference_dtype";
    }
}

GLVUnet2dConditionalModelOp::~GLVUnet2dConditionalModelOp()
{
    delete model_;
}

th::Tensor GLVUnet2dConditionalModelOp::forward(th::Tensor                                      hidden_states,
                                                th::Tensor                                      encoder_hidden_states,
                                                const double                                    timestep,
                                                th::Tensor&                                     aug_emb,
                                                th::optional<vector<std::string>>               controlnet_names,
                                                th::optional<vector<th::Tensor>>                controlnet_conds,
                                                th::optional<vector<th::Tensor>>                controlnet_augs,
                                                th::optional<vector<vector<double>>>            controlnet_scales,
                                                th::optional<bool>                              controlnet_guess_mode,
                                                th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                                th::optional<th::Dict<std::string, double>>     scale_params)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_TH_CUDA(aug_emb);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    CHECK_CONTIGUOUS(aug_emb);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
        TORCH_CHECK(aug_emb.dtype() == torch::kFloat32, "aug_emb dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
        TORCH_CHECK(aug_emb.dtype() == torch::kFloat16, "aug_emb dtype should be float16");
    }

    return model_->forward(hidden_states,
                           encoder_hidden_states,
                           timestep,
                           aug_emb,
                           controlnet_names,
                           controlnet_conds,
                           controlnet_augs,
                           controlnet_scales,
                           controlnet_guess_mode,
                           map_extra_tensors,
                           scale_params);
}

void GLVUnet2dConditionalModelOp::load_lora(const std::string lora_path,
                                            const std::string lora_name,
                                            const double      lora_strength,
                                            const std::string model_dtype)
{
    model_->load_lora(lora_path, lora_name, lora_strength, model_dtype);
}

void GLVUnet2dConditionalModelOp::unload_lora(const std::string lora_name, const bool clean_mem_cache)
{
    model_->unload_lora(lora_name, clean_mem_cache);
}

void GLVUnet2dConditionalModelOp::reload_unet_model(const std::string model_path, const std::string model_dtype)
{
    model_->reload_unet_model(model_path, model_dtype);
}

void GLVUnet2dConditionalModelOp::reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights,
                                                               const std::string                 device)
{
    model_->reload_unet_model_from_cache(weights, device);
}

void GLVUnet2dConditionalModelOp::load_lora_by_name(std::vector<std::string> lora_name,
                                                    th::Tensor               lora_weights,
                                                    const double             lora_strength)
{
    model_->load_lora_by_name(lora_name, lora_weights, lora_strength);
}

void GLVUnet2dConditionalModelOp::load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                            const double                      lora_strength)
{
    model_->load_lora_from_state_dict(weights, lora_strength);
}

void GLVUnet2dConditionalModelOp::load_controlnet_model(const std::string controlnet_name,
                                                        const std::string model_path,
                                                        const std::string model_dtype)
{
    model_->load_controlnet_model(controlnet_name, model_path, model_dtype);
}

void GLVUnet2dConditionalModelOp::unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache)
{
    model_->unload_controlnet_model(controlnet_name, clean_mem_cache);
}

void GLVUnet2dConditionalModelOp::load_controlnet_model_from_state_dict(const std::string controlnet_name,
                                                                        th::Dict<std::string, th::Tensor> weights,
                                                                        const std::string                 device)
{
    model_->load_controlnet_model_from_state_dict(controlnet_name, weights, device);
}

void GLVUnet2dConditionalModelOp::clean_controlnet_cache()
{
    model_->clean_controlnet_cache();
}

void GLVUnet2dConditionalModelOp::clean_lora_cache()
{
    model_->clean_lora_cache();
}

std::vector<std::string> GLVUnet2dConditionalModelOp::get_loaded_controlnet()
{
    return model_->get_loaded_controlnet();
}

std::vector<std::string> GLVUnet2dConditionalModelOp::get_loaded_lora()
{
    return model_->get_loaded_lora();
}

void GLVUnet2dConditionalModelOp::load_ip_adapter(const std::string& ip_adapter_path,
                                                  const std::string& ip_adapter_name,
                                                  const double&      scale,
                                                  const std::string& model_dtype)
{
    model_->load_ip_adapter(ip_adapter_path, ip_adapter_name, scale, model_dtype);
}

void GLVUnet2dConditionalModelOp::unload_ip_adapter(const std::string& ip_adapter_name)
{
    model_->unload_ip_adapter(ip_adapter_name);
}

}  // namespace torch_ext

static auto lyradiffGLVUnet2dConditionalModelTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GLVUnet2dConditionalModelOp>("lyradiffGLVUnet2dConditionalModelOp")
#else
    torch::jit::class_<torch_ext::GLVUnet2dConditionalModelOp>("lyradiff", "GLVUnet2dConditionalModelOp")
#endif
        .def(torch::jit::init<std::string, int64_t, int64_t>(),
             "init",
             {torch::arg("inference_dtype") = "fp16",
              torch::arg("input_channels")  = 4,
              torch::arg("output_channels") = 4})
        .def("forward",
             &torch_ext::GLVUnet2dConditionalModelOp::forward,
             "forward",
             {torch::arg("hidden_states"),
              torch::arg("encoder_hidden_states"),
              torch::arg("timestep"),
              torch::arg("aug_emb"),
              torch::arg("controlnet_names")      = th::nullopt,
              torch::arg("controlnet_conds")      = th::nullopt,
              torch::arg("controlnet_augs")       = th::nullopt,
              torch::arg("controlnet_scales")     = th::nullopt,
              torch::arg("controlnet_guess_mode") = th::nullopt,
              torch::arg("map_extra_tensors")     = th::nullopt,
              torch::arg("scale_params")          = th::nullopt})
        .def("load_lora", &torch_ext::GLVUnet2dConditionalModelOp::load_lora)
        .def("load_lora_from_state_dict", &torch_ext::GLVUnet2dConditionalModelOp::load_lora_from_state_dict)
        .def("unload_lora", &torch_ext::GLVUnet2dConditionalModelOp::unload_lora)
        .def("reload_unet_model", &torch_ext::GLVUnet2dConditionalModelOp::reload_unet_model)
        .def("load_controlnet_model", &torch_ext::GLVUnet2dConditionalModelOp::load_controlnet_model)
        .def("load_controlnet_model_from_state_dict",
             &torch_ext::GLVUnet2dConditionalModelOp::load_controlnet_model_from_state_dict)
        .def("load_lora_by_name", &torch_ext::GLVUnet2dConditionalModelOp::load_lora_by_name)
        .def("unload_controlnet_model", &torch_ext::GLVUnet2dConditionalModelOp::unload_controlnet_model)
        .def("reload_unet_model_from_cache", &torch_ext::GLVUnet2dConditionalModelOp::reload_unet_model_from_cache)
        .def("get_loaded_controlnet", &torch_ext::GLVUnet2dConditionalModelOp::get_loaded_controlnet)
        .def("clean_lora_cache", &torch_ext::GLVUnet2dConditionalModelOp::clean_lora_cache)
        .def("clean_controlnet_cache", &torch_ext::GLVUnet2dConditionalModelOp::clean_controlnet_cache)
        .def("get_loaded_lora", &torch_ext::GLVUnet2dConditionalModelOp::get_loaded_lora)
        .def("load_ip_adapter", &torch_ext::GLVUnet2dConditionalModelOp::load_ip_adapter)
        .def("unload_ip_adapter", &torch_ext::GLVUnet2dConditionalModelOp::unload_ip_adapter);
