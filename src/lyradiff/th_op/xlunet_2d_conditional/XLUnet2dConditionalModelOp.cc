#include "src/lyradiff/th_op/xlunet_2d_conditional/XLUnet2dConditionalModelOp.h"

namespace th = torch;
namespace torch_ext {

XLUnet2dConditionalModelOp::XLUnet2dConditionalModelOp(const std::string inference_dtype,
                                                       const int64_t     input_channels,
                                                       const int64_t     output_channels,
                                                       const int64_t     quant_level):
    inference_dtype_(inference_dtype)
{

    if (inference_dtype_ == "fp32") {
        model_ =
            new LyraDiffXLUnet2dConditionalModel<float>(inference_dtype, input_channels, output_channels, quant_level);
    }
    else if (inference_dtype_ == "fp16") {
        model_ =
            new LyraDiffXLUnet2dConditionalModel<half>(inference_dtype, input_channels, output_channels, quant_level);
    }
    else {
        throw "wrong inference_dtype";
    }
}

XLUnet2dConditionalModelOp::~XLUnet2dConditionalModelOp()
{
    delete model_;
}

th::Tensor XLUnet2dConditionalModelOp::forward(th::Tensor                                      hidden_states,
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

th::Tensor XLUnet2dConditionalModelOp::unet_forward(th::Tensor                       hidden_states,
                                                    th::Tensor                       encoder_hidden_states,
                                                    th::Tensor                       aug_emb,
                                                    const double                     timestep,
                                                    th::optional<vector<th::Tensor>> controlnet_results,
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

    return model_->unet_forward(
        hidden_states, encoder_hidden_states, aug_emb, timestep, controlnet_results, map_extra_tensors, scale_params);
}

std::vector<th::Tensor> XLUnet2dConditionalModelOp::controlnet_forward(th::Tensor          hidden_states,
                                                                       th::Tensor          encoder_hidden_states,
                                                                       th::Tensor          conditioning_img,
                                                                       th::Tensor          controlnet_aug,
                                                                       const double        timestep,
                                                                       std::vector<double> controlnet_scales,
                                                                       std::string         controlnet_name)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_TH_CUDA(conditioning_img);
    CHECK_TH_CUDA(controlnet_aug);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    CHECK_CONTIGUOUS(conditioning_img);
    CHECK_CONTIGUOUS(controlnet_aug);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
        TORCH_CHECK(conditioning_img.dtype() == torch::kFloat32, "conditioning_img dtype should be float32");
        TORCH_CHECK(controlnet_aug.dtype() == torch::kFloat32, "controlnet_aug dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
        TORCH_CHECK(conditioning_img.dtype() == torch::kFloat16, "conditioning_img dtype should be float16");
        TORCH_CHECK(controlnet_aug.dtype() == torch::kFloat16, "controlnet_aug dtype should be float16");
    }

    return model_->controlnet_forward(hidden_states,
                                      encoder_hidden_states,
                                      conditioning_img,
                                      controlnet_aug,
                                      timestep,
                                      controlnet_scales,
                                      controlnet_name);
}

void XLUnet2dConditionalModelOp::load_lora(const std::string lora_path,
                                           const std::string lora_name,
                                           const double      lora_strength,
                                           const std::string model_dtype)
{
    model_->load_lora(lora_path, lora_name, lora_strength, model_dtype);
}

void XLUnet2dConditionalModelOp::unload_lora(const std::string lora_name, const bool clean_mem_cache)
{
    model_->unload_lora(lora_name, clean_mem_cache);
}

void XLUnet2dConditionalModelOp::reload_unet_model(const std::string model_path, const std::string model_dtype)
{
    model_->reload_unet_model(model_path, model_dtype);
}

void XLUnet2dConditionalModelOp::reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights,
                                                              const std::string                 device)
{
    model_->reload_unet_model_from_cache(weights, device);
}

void XLUnet2dConditionalModelOp::load_lora_by_name(std::vector<std::string> lora_name,
                                                   th::Tensor               lora_weights,
                                                   const double             lora_strength)
{
    model_->load_lora_by_name(lora_name, lora_weights, lora_strength);
}

void XLUnet2dConditionalModelOp::load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                           const double                      lora_strength)
{
    model_->load_lora_from_state_dict(weights, lora_strength);
}

void XLUnet2dConditionalModelOp::load_controlnet_model(const std::string controlnet_name,
                                                       const std::string model_path,
                                                       const std::string model_dtype,
                                                       const std::string controlnet_mode)
{
    model_->load_controlnet_model(controlnet_name, model_path, model_dtype, controlnet_mode);
}

void XLUnet2dConditionalModelOp::unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache)
{
    model_->unload_controlnet_model(controlnet_name, clean_mem_cache);
}

void XLUnet2dConditionalModelOp::load_controlnet_model_from_state_dict(const std::string controlnet_name,
                                                                       th::Dict<std::string, th::Tensor> weights,
                                                                       const std::string                 device,
                                                                       const std::string controlnet_mode)
{
    model_->load_controlnet_model_from_state_dict(controlnet_name, weights, device, controlnet_mode);
}

void XLUnet2dConditionalModelOp::clean_controlnet_cache()
{
    model_->clean_controlnet_cache();
}

void XLUnet2dConditionalModelOp::clean_lora_cache()
{
    model_->clean_lora_cache();
}

std::vector<std::string> XLUnet2dConditionalModelOp::get_loaded_controlnet()
{
    return model_->get_loaded_controlnet();
}

std::vector<std::string> XLUnet2dConditionalModelOp::get_loaded_lora()
{
    return model_->get_loaded_lora();
}

void XLUnet2dConditionalModelOp::load_ip_adapter(const std::string& ip_adapter_path,
                                                 const std::string& ip_adapter_name,
                                                 const double&      scale,
                                                 const std::string& model_dtype)
{
    model_->load_ip_adapter(ip_adapter_path, ip_adapter_name, scale, model_dtype);
}

void XLUnet2dConditionalModelOp::unload_ip_adapter(const std::string& ip_adapter_name)
{
    model_->unload_ip_adapter(ip_adapter_name);
}

}  // namespace torch_ext

// .def("forward",
//      &torch_ext::Unet2dConditionalModelOp::forward,
//      "forward",
//      {torch::arg("hidden_states"),
//       torch::arg("encoder_hidden_states"),
//       torch::arg("timestep")              = 0,
//       torch::arg("controlnet_names")      = th::nullopt,
//       torch::arg("controlnet_conds")      = th::nullopt,
//       torch::arg("controlnet_scales")     = th::nullopt,
//       torch::arg("controlnet_guess_mode") = th::nullopt,
//       torch::arg("map_extra_tensors")     = th::nullopt,
//       torch::arg("scale_params")          = th::nullopt})

static auto LyraDiffXLUnet2dConditionalModelTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::XLUnet2dConditionalModelOp>("LyraDiffXLUnet2dConditionalModelOp")
#else
    torch::jit::class_<torch_ext::XLUnet2dConditionalModelOp>("lyradiff", "XLUnet2dConditionalModelOp")
#endif
        .def(torch::jit::init<std::string, int64_t, int64_t, int64_t>(),
             "init",
             {torch::arg("inference_dtype") = "fp16",
              torch::arg("input_channels")  = 4,
              torch::arg("output_channels") = 4,
              torch::arg("quant_level")     = 0})
        .def("forward",
             &torch_ext::XLUnet2dConditionalModelOp::forward,
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
        .def("controlnet_forward", &torch_ext::XLUnet2dConditionalModelOp::controlnet_forward)
        .def("unet_forward",
             &torch_ext::XLUnet2dConditionalModelOp::unet_forward,
             "unet_forward",
             {torch::arg("hidden_states"),
              torch::arg("encoder_hidden_states"),
              torch::arg("aug_emb"),
              torch::arg("timestep"),
              torch::arg("controlnet_results") = th::nullopt,
              torch::arg("map_extra_tensors")  = th::nullopt,
              torch::arg("scale_params")       = th::nullopt})
        .def("load_lora", &torch_ext::XLUnet2dConditionalModelOp::load_lora)
        .def("load_lora_from_state_dict", &torch_ext::XLUnet2dConditionalModelOp::load_lora_from_state_dict)
        .def("unload_lora", &torch_ext::XLUnet2dConditionalModelOp::unload_lora)
        .def("reload_unet_model", &torch_ext::XLUnet2dConditionalModelOp::reload_unet_model)
        // .def("load_controlnet_model", &torch_ext::XLUnet2dConditionalModelOp::load_controlnet_model)
        // .def("load_controlnet_model_from_state_dict",
        //      &torch_ext::XLUnet2dConditionalModelOp::load_controlnet_model_from_state_dict)
        .def("load_controlnet_model",
             &torch_ext::XLUnet2dConditionalModelOp::load_controlnet_model,
             "load_controlnet_model",
             {torch::arg("controlnet_name"),
              torch::arg("model_path"),
              torch::arg("model_dtype"),
              torch::arg("controlnet_mode") = "large"})
        .def("load_controlnet_model_from_state_dict",
             &torch_ext::XLUnet2dConditionalModelOp::load_controlnet_model_from_state_dict,
             "load_controlnet_model_from_state_dict",
             {torch::arg("controlnet_name"),
              torch::arg("weights"),
              torch::arg("device"),
              torch::arg("controlnet_mode") = "large"})
        .def("load_lora_by_name", &torch_ext::XLUnet2dConditionalModelOp::load_lora_by_name)
        .def("unload_controlnet_model", &torch_ext::XLUnet2dConditionalModelOp::unload_controlnet_model)
        .def("reload_unet_model_from_cache", &torch_ext::XLUnet2dConditionalModelOp::reload_unet_model_from_cache)
        .def("get_loaded_controlnet", &torch_ext::XLUnet2dConditionalModelOp::get_loaded_controlnet)
        .def("clean_lora_cache", &torch_ext::XLUnet2dConditionalModelOp::clean_lora_cache)
        .def("clean_controlnet_cache", &torch_ext::XLUnet2dConditionalModelOp::clean_controlnet_cache)
        .def("get_loaded_lora", &torch_ext::XLUnet2dConditionalModelOp::get_loaded_lora)
        .def("load_ip_adapter", &torch_ext::XLUnet2dConditionalModelOp::load_ip_adapter)
        .def("unload_ip_adapter", &torch_ext::XLUnet2dConditionalModelOp::unload_ip_adapter);
