#pragma once

#include "src/lyradiff/th_op/flux_controlnet_model/FluxControlnetModelOp.h"
#include "src/lyradiff/th_op/flux_single_transformer_block/FluxSingleTransformerBlockOp.h"
#include "src/lyradiff/th_op/flux_transformer_2d_model/FluxTransformer2DModelOp.h"
#include "src/lyradiff/th_op/flux_transformer_block/FluxTransformerBlockOp.h"
#include "src/lyradiff/th_op/lyradiff_common_context/LyraDiffCommonContext.h"

TORCH_LIBRARY(lyradiff, m)
{
    m.class_<torch_ext::LyraDiffCommonContext>("LyraDiffCommonContext").def(torch::jit::init<>(), "init", {});

    m.class_<torch_ext::FluxTransformer2DModelOp>("FluxTransformer2DModelOp")
        .def(torch::jit::
                 init<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t>(),
             "init",
             {torch::arg("inference_dtype")       = "bf16",
              torch::arg("input_channels")        = 64,
              torch::arg("num_layers")            = 19,
              torch::arg("num_single_layers")     = 38,
              torch::arg("attention_head_dim")    = 128,
              torch::arg("num_attention_heads")   = 24,
              torch::arg("pooled_projection_dim") = 768,
              torch::arg("joint_attention_dim")   = 4096,
              torch::arg("guidance_embeds")       = true,
              torch::arg("quant_level")           = 0})
        .def("initialize", &torch_ext::FluxTransformer2DModelOp::initialize)
        .def("transformer_forward",
             &torch_ext::FluxTransformer2DModelOp::transformer_forward,
             "transformer_forward",
             {torch::arg("hidden_states"),
              torch::arg("encoder_hidden_states"),
              torch::arg("rope_emb"),
              torch::arg("pooled_projection"),
              torch::arg("timestep"),
              torch::arg("guidance"),
              torch::arg("controlnet_block_samples")        = th::nullopt,
              torch::arg("controlnet_single_block_samples") = th::nullopt,
              torch::arg("controlnet_blocks_repeat")        = false})
        .def("reload_transformer_model", &torch_ext::FluxTransformer2DModelOp::reload_transformer_model)
        .def("load_lora_by_name", &torch_ext::FluxTransformer2DModelOp::load_lora_by_name)
        .def("clear_lora", &torch_ext::FluxTransformer2DModelOp::clear_lora)
        .def("reload_transformer_model_from_state_dict",
             &torch_ext::FluxTransformer2DModelOp::reload_transformer_model_from_state_dict);

    m.class_<torch_ext::FluxControlnetModelOp>("FluxControlnetModelOp")
        .def(torch::jit::
                 init<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t>(),
             "init",
             {torch::arg("inference_dtype")       = "bf16",
              torch::arg("input_channels")        = 64,
              torch::arg("num_layers")            = 2,
              torch::arg("num_single_layers")     = 0,
              torch::arg("attention_head_dim")    = 128,
              torch::arg("num_attention_heads")   = 24,
              torch::arg("pooled_projection_dim") = 768,
              torch::arg("joint_attention_dim")   = 4096,
              torch::arg("guidance_embeds")       = true,
              torch::arg("quant_level")           = 0})
        .def("initialize", &torch_ext::FluxControlnetModelOp::initialize)
        .def("controlnet_forward", &torch_ext::FluxControlnetModelOp::controlnet_forward)
        .def("reload_controlnet_model", &torch_ext::FluxControlnetModelOp::reload_controlnet_model)
        .def("reload_controlnet_model_from_state_dict",
             &torch_ext::FluxControlnetModelOp::reload_controlnet_model_from_state_dict);

    m.class_<torch_ext::FluxTransformerBlockOp>("FluxTransformerBlockOp")
        .def(torch::jit::init<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>(),
             "init",
             {torch::arg("inference_dtype")    = "bf16",
              torch::arg("embedding_dim")      = 3072,
              torch::arg("embedding_head_num") = 24,
              torch::arg("embedding_head_dim") = 128,
              torch::arg("mlp_scale")          = 6,
              torch::arg("quant_level")        = 0})
        .def("initialize", &torch_ext::FluxTransformerBlockOp::initialize)
        .def("forward", &torch_ext::FluxTransformerBlockOp::forward)
        .def("load_lora_by_name", &torch_ext::FluxTransformerBlockOp::load_lora_by_name)
        .def("clear_lora", &torch_ext::FluxTransformerBlockOp::clear_lora)
        .def("reload_model_from_bin", &torch_ext::FluxTransformerBlockOp::reload_model_from_bin)
        .def("reload_model_from_state_dict", &torch_ext::FluxTransformerBlockOp::reload_model_from_state_dict);

    m.class_<torch_ext::FluxSingleTransformerBlockOp>("FluxSingleTransformerBlockOp")
        .def(torch::jit::init<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>(),
             "init",
             {torch::arg("inference_dtype")    = "bf16",
              torch::arg("embedding_dim")      = 3072,
              torch::arg("embedding_head_num") = 24,
              torch::arg("embedding_head_dim") = 128,
              torch::arg("mlp_scale")          = 4,
              torch::arg("quant_level")        = 0})
        .def("initialize", &torch_ext::FluxSingleTransformerBlockOp::initialize)
        .def("forward", &torch_ext::FluxSingleTransformerBlockOp::forward)
        .def("load_lora_by_name", &torch_ext::FluxSingleTransformerBlockOp::load_lora_by_name)
        .def("clear_lora", &torch_ext::FluxSingleTransformerBlockOp::clear_lora)
        .def("reload_model_from_bin", &torch_ext::FluxSingleTransformerBlockOp::reload_model_from_bin)
        .def("reload_model_from_state_dict", &torch_ext::FluxSingleTransformerBlockOp::reload_model_from_state_dict);
}