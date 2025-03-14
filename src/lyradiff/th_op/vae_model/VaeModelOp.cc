#include "VaeModelOp.h"

namespace th = torch;
namespace torch_ext {

VaeModelOp::VaeModelOp(const std::string inference_dtype, const bool is_upcast): inference_dtype_(inference_dtype)
{

    if (inference_dtype_ == "fp32") {
        model_ = new LyraDiffVaeModel<float>(inference_dtype, false);
    }
    else if (inference_dtype_ == "fp16") {
        model_ = new LyraDiffVaeModel<half>(inference_dtype, is_upcast);
    }
    else {
        throw "wrong inference_dtype";
    }
}

VaeModelOp::~VaeModelOp()
{
    delete model_;
}

th::Tensor VaeModelOp::forward(th::Tensor hidden_states)
{
    model_->forward(hidden_states);
}

void VaeModelOp::reload_vae_model(const std::string model_path, const std::string model_dtype)
{
    model_->reload_vae_model(model_path, model_dtype);
}

void VaeModelOp::reload_vae_encoder(const std::string model_path, const std::string model_dtype)
{
    model_->reload_vae_encoder(model_path, model_dtype);
}

void VaeModelOp::reload_vae_decoder(const std::string model_path, const std::string model_dtype)
{
    model_->reload_vae_decoder(model_path, model_dtype);
}

void VaeModelOp::reload_vae_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
{
    weight_loader_manager_glob->cur_load_module = "vae";
    return model_->reload_vae_model_from_cache(weights, device);
}

void VaeModelOp::reload_vae_encoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
{
    return model_->reload_vae_encoder_from_cache(weights, device);
}

void VaeModelOp::reload_vae_decoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
{
    return model_->reload_vae_decoder_from_cache(weights, device);
}

void VaeModelOp::load_s3diff_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights_alpha,
                                                  th::Dict<std::string, th::Tensor> weights_beta)
{
    return model_->load_s3diff_lora_from_state_dict(weights_alpha, weights_beta);
}

th::Tensor VaeModelOp::vae_decode(th::Tensor                                      hidden_states,
                                  th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                  th::optional<th::Dict<std::string, double>>     scale_params)
{
    return model_->vae_decode(hidden_states, map_extra_tensors, scale_params);
}

th::Tensor VaeModelOp::vae_encode(th::Tensor                                      hidden_states,
                                  th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                  th::optional<th::Dict<std::string, double>>     scale_params)
{
    return model_->vae_encode(hidden_states, map_extra_tensors, scale_params);
}

}  // namespace torch_ext

static auto LyraDiffVaeModelTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::VaeModelOp>("VaeModelOp")
#else
    torch::jit::class_<torch_ext::VaeModelOp>("lyradiff", "VaeModelOp")
#endif
        .def(torch::jit::init<std::string, bool>(),
             "init",
             {torch::arg("inference_dtype") = "fp16", torch::arg("is_upcast") = false})
        .def("forward", &torch_ext::VaeModelOp::forward)
        .def("reload_vae_model", &torch_ext::VaeModelOp::reload_vae_model)
        .def("reload_vae_encoder", &torch_ext::VaeModelOp::reload_vae_encoder)
        .def("reload_vae_decoder", &torch_ext::VaeModelOp::reload_vae_decoder)
        .def("reload_vae_model_from_cache", &torch_ext::VaeModelOp::reload_vae_model_from_cache)
        .def("reload_vae_encoder_from_cache", &torch_ext::VaeModelOp::reload_vae_encoder_from_cache)
        .def("reload_vae_decoder_from_cache", &torch_ext::VaeModelOp::reload_vae_decoder_from_cache)
        .def("load_s3diff_lora_from_state_dict", &torch_ext::VaeModelOp::load_s3diff_lora_from_state_dict)
        .def("vae_encode",
             &torch_ext::VaeModelOp::vae_encode,
             "vae_encode",
             {
                 torch::arg("hidden_states"),
                 torch::arg("map_extra_tensors") = th::nullopt,
                 torch::arg("scale_params")      = th::nullopt,
             })
        .def("vae_decode",
             &torch_ext::VaeModelOp::vae_decode,
             "vae_decode",
             {
                 torch::arg("hidden_states"),
                 torch::arg("map_extra_tensors") = th::nullopt,
                 torch::arg("scale_params")      = th::nullopt,
             });
