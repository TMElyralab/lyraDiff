#include "src/lyradiff/th_op/unet_2d_conditional/Unet2dConditionalModelOp.h"

namespace th = torch;
namespace py = pybind11;

namespace torch_ext {

Unet2dConditionalModelOp::Unet2dConditionalModelOp(const int64_t             max_controlnet_count,
                                                   const std::string         inference_dtype,
                                                   th::optional<int64_t>     input_channels,
                                                   th::optional<int64_t>     output_channels,
                                                   th::optional<int64_t>     quant_level,
                                                   th::optional<std::string> sd_ver):
    inference_dtype_(inference_dtype)
{
    size_t input_channels_  = input_channels.has_value() ? input_channels.value() : 4;
    size_t output_channels_ = output_channels.has_value() ? output_channels.value() : 4;
    size_t quant_level_     = quant_level.has_value() ? quant_level.value() : 0;

    std::string sd_ver_ = sd_ver.has_value() ? sd_ver.value() : "sd15";

    if (inference_dtype_ == "fp32") {
        model_ = new LyraDiffUnet2dConditionalModel<float>(
            max_controlnet_count, inference_dtype, input_channels_, output_channels_, quant_level_, sd_ver_);
    }
    else if (inference_dtype_ == "fp16") {
        model_ = new LyraDiffUnet2dConditionalModel<half>(
            max_controlnet_count, inference_dtype, input_channels_, output_channels_, quant_level_, sd_ver_);
    }
    else {
        throw "wrong inference_dtype";
    }
}

Unet2dConditionalModelOp::~Unet2dConditionalModelOp()
{
    delete model_;
}

th::Tensor Unet2dConditionalModelOp::forward(th::Tensor                                      hidden_states,
                                             th::Tensor                                      encoder_hidden_states,
                                             const double                                    timestep,
                                             th::optional<vector<std::string>>               controlnet_names,
                                             th::optional<vector<th::Tensor>>                controlnet_conds,
                                             th::optional<vector<vector<double>>>            controlnet_scales,
                                             th::optional<bool>                              controlnet_guess_mode,
                                             th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                             th::optional<th::Dict<std::string, double>>     scale_params)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
    }

    return model_->forward(hidden_states,
                           encoder_hidden_states,
                           timestep,
                           controlnet_names,
                           controlnet_conds,
                           controlnet_scales,
                           controlnet_guess_mode,
                           map_extra_tensors,
                           //    ip_hidden_states,
                           //    fp_hidden_states,
                           scale_params);
}

std::vector<th::Tensor> Unet2dConditionalModelOp::controlnet_forward(th::Tensor          hidden_states,
                                                                     th::Tensor          encoder_hidden_states,
                                                                     th::Tensor          conditioning_img,
                                                                     const double        timestep,
                                                                     std::vector<double> controlnet_scales,
                                                                     std::string         controlnet_name)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_TH_CUDA(conditioning_img);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    CHECK_CONTIGUOUS(conditioning_img);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
        TORCH_CHECK(conditioning_img.dtype() == torch::kFloat32, "conditioning_img dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
        TORCH_CHECK(conditioning_img.dtype() == torch::kFloat16, "conditioning_img dtype should be float16");
    }

    return model_->controlnet_forward(
        hidden_states, encoder_hidden_states, conditioning_img, timestep, controlnet_scales, controlnet_name);
}

th::Tensor Unet2dConditionalModelOp::unet_forward(th::Tensor                                      hidden_states,
                                                  th::Tensor                                      encoder_hidden_states,
                                                  const double                                    timestep,
                                                  th::optional<vector<th::Tensor>>                controlnet_results,
                                                  th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                                  th::optional<th::Dict<std::string, double>>     scale_params)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(encoder_hidden_states);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(encoder_hidden_states);
    if (inference_dtype_ == "fp32") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat32, "hidden_states dtype should be float32");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat32, "encoder_hidden_states dtype should be float32");
    }
    else if (inference_dtype_ == "fp16") {
        TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states dtype should be float16");
        TORCH_CHECK(encoder_hidden_states.dtype() == torch::kFloat16, "encoder_hidden_states dtype should be float16");
    }

    return model_->unet_forward(
        hidden_states, encoder_hidden_states, timestep, controlnet_results, map_extra_tensors, scale_params);
}

void Unet2dConditionalModelOp::load_lora(const std::string lora_path,
                                         const std::string lora_name,
                                         const double      lora_strength,
                                         const std::string model_dtype)
{
    model_->load_lora(lora_path, lora_name, lora_strength, model_dtype);
}

void Unet2dConditionalModelOp::unload_lora(const std::string lora_name, const bool clean_mem_cache)
{
    model_->unload_lora(lora_name, clean_mem_cache);
}

void Unet2dConditionalModelOp::reload_unet_model(const std::string model_path, const std::string model_dtype)
{
    model_->reload_unet_model(model_path, model_dtype);
}

void Unet2dConditionalModelOp::reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights,
                                                            const std::string                 device)
{
    weight_loader_manager_glob->set_cur_load_module("unet");
    model_->reload_unet_model_from_cache(weights, device);
}

void Unet2dConditionalModelOp::load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                         const double                      lora_strength)
{
    model_->load_lora_from_state_dict(weights, lora_strength);
}

void Unet2dConditionalModelOp::load_s3diff_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights_alpha,
                                                                th::Dict<std::string, th::Tensor> weights_beta)
{
    model_->load_s3diff_lora_from_state_dict(weights_alpha, weights_beta);
}

void Unet2dConditionalModelOp::load_lora_by_name(std::vector<std::string> lora_name,
                                                 th::Tensor               lora_weights,
                                                 const double             lora_strength)
{
    model_->load_lora_by_name(lora_name, lora_weights, lora_strength);
}

void Unet2dConditionalModelOp::load_controlnet_model(const std::string controlnet_name,
                                                     const std::string model_path,
                                                     const std::string model_dtype)
{
    model_->load_controlnet_model(controlnet_name, model_path, model_dtype);
}

void Unet2dConditionalModelOp::unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache)
{
    model_->unload_controlnet_model(controlnet_name, clean_mem_cache);
}

void Unet2dConditionalModelOp::load_controlnet_model_from_state_dict(const std::string                 controlnet_name,
                                                                     th::Dict<std::string, th::Tensor> weights,
                                                                     const std::string                 device)
{
    model_->load_controlnet_model_from_state_dict(controlnet_name, weights, device);
}

void Unet2dConditionalModelOp::clean_controlnet_cache()
{
    model_->clean_controlnet_cache();
}

void Unet2dConditionalModelOp::free_buffer()
{
    model_->free_buffer();
}

void Unet2dConditionalModelOp::clean_lora_cache()
{
    model_->clean_lora_cache();
}

std::vector<std::string> Unet2dConditionalModelOp::get_loaded_controlnet()
{
    return model_->get_loaded_controlnet();
}

std::vector<std::string> Unet2dConditionalModelOp::get_loaded_lora()
{
    return model_->get_loaded_lora();
}

void Unet2dConditionalModelOp::load_ip_adapter(const std::string& ip_adapter_path,
                                               const std::string& ip_adapter_name,
                                               const double&      scale,
                                               const std::string& model_dtype)
{
    model_->load_ip_adapter(ip_adapter_path, ip_adapter_name, scale, model_dtype);
}

void Unet2dConditionalModelOp::unload_ip_adapter(const std::string& ip_adapter_name)
{
    model_->unload_ip_adapter(ip_adapter_name);
}

}  // namespace torch_ext

static auto LyraDiffUnet2dConditionalModelTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::Unet2dConditionalModelOp>("LyraDiffUnet2dConditionalModelOp")
#else
    torch::jit::class_<torch_ext::Unet2dConditionalModelOp>("lyradiff", "Unet2dConditionalModelOp")
#endif
        .def(torch::jit::init<int64_t,
                              std::string,
                              th::optional<int64_t>,
                              th::optional<int64_t>,
                              th::optional<int64_t>,
                              th::optional<std::string>>(),
             "init",
             {torch::arg("max_controlnet_count") = 3,
              torch::arg("inference_dtype")      = "fp16",
              torch::arg("input_channels")       = 4,
              torch::arg("output_channels")      = 4,
              torch::arg("quant_level")          = 0,
              torch::arg("sd_ver")               = "sd15"})
        .def("forward",
             &torch_ext::Unet2dConditionalModelOp::forward,
             "forward",
             {torch::arg("hidden_states"),
              torch::arg("encoder_hidden_states"),
              torch::arg("timestep")              = 0,
              torch::arg("controlnet_names")      = th::nullopt,
              torch::arg("controlnet_conds")      = th::nullopt,
              torch::arg("controlnet_scales")     = th::nullopt,
              torch::arg("controlnet_guess_mode") = th::nullopt,
              torch::arg("map_extra_tensors")     = th::nullopt,
              torch::arg("scale_params")          = th::nullopt})
        .def("controlnet_forward", &torch_ext::Unet2dConditionalModelOp::controlnet_forward)
        .def("unet_forward",
             &torch_ext::Unet2dConditionalModelOp::unet_forward,
             "unet_forward",
             {torch::arg("hidden_states"),
              torch::arg("encoder_hidden_states"),
              torch::arg("timestep")           = 0,
              torch::arg("controlnet_results") = th::nullopt,
              torch::arg("map_extra_tensors")  = th::nullopt,
              torch::arg("scale_params")       = th::nullopt})
        .def("load_lora", &torch_ext::Unet2dConditionalModelOp::load_lora)
        .def("load_lora_from_state_dict", &torch_ext::Unet2dConditionalModelOp::load_lora_from_state_dict)
        .def("load_s3diff_lora_from_state_dict", &torch_ext::Unet2dConditionalModelOp::load_s3diff_lora_from_state_dict)
        .def("load_lora_by_name", &torch_ext::Unet2dConditionalModelOp::load_lora_by_name)
        .def("unload_lora", &torch_ext::Unet2dConditionalModelOp::unload_lora)
        .def("reload_unet_model", &torch_ext::Unet2dConditionalModelOp::reload_unet_model)
        .def("load_controlnet_model", &torch_ext::Unet2dConditionalModelOp::load_controlnet_model)
        .def("load_controlnet_model_from_state_dict",
             &torch_ext::Unet2dConditionalModelOp::load_controlnet_model_from_state_dict)
        .def("unload_controlnet_model", &torch_ext::Unet2dConditionalModelOp::unload_controlnet_model)
        .def("reload_unet_model_from_cache", &torch_ext::Unet2dConditionalModelOp::reload_unet_model_from_cache)
        .def("get_loaded_controlnet", &torch_ext::Unet2dConditionalModelOp::get_loaded_controlnet)
        .def("clean_lora_cache", &torch_ext::Unet2dConditionalModelOp::clean_lora_cache)
        .def("clean_controlnet_cache", &torch_ext::Unet2dConditionalModelOp::clean_controlnet_cache)
        .def("get_loaded_lora", &torch_ext::Unet2dConditionalModelOp::get_loaded_lora)
        .def("free_buffer", &torch_ext::Unet2dConditionalModelOp::free_buffer)
        .def("load_ip_adapter", &torch_ext::Unet2dConditionalModelOp::load_ip_adapter)
        .def("unload_ip_adapter", &torch_ext::Unet2dConditionalModelOp::unload_ip_adapter);
