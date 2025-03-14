import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import gc
import torch
import numpy as np
from glob import glob

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler,
                                  KarrasDiffusionSchedulers)
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from .lyradiff_vae_model import LyraDiffVaeModel
from .module.lyradiff_ip_adapter import LyraIPAdapter
from .lora_util import add_text_lora_layer, add_xltext_lora_layer, add_lora_to_opt_model, load_lora_state_dict
from safetensors.torch import load_file
from .lyradiff_unet_model import LyraDiffUNet2DConditionModel
from .lyradiff_controlnet_model import LyraDiffControlNetModel


class LyraDiffPipelineBase(TextualInversionLoaderMixin):
    _optional_components = ["safety_checker",
                            "feature_extractor", "image_encoder"]

    def __init__(self, device=torch.device("cuda"), dtype=torch.float16, num_channels_unet=4, num_channels_latents=4, vae_scale_factor=8, vae_scaling_factor=0.18215, split_controlnet=False, quant_level=0) -> None:
        self.device = device
        self.dtype = dtype

        self.quant_level = quant_level

        self.num_channels_unet = num_channels_unet
        self.num_channels_latents = num_channels_latents
        self.vae_scale_factor = vae_scale_factor
        self.vae_scaling_factor = vae_scaling_factor

        self.unet_cache = {}
        self.unet_in_channels = 4

        self.text_encoder_cache = {}
        self.tokenizer_cache = {}

        self.controlnet_cache = {}

        self.loaded_lora = {}
        self.loaded_lora_strength = {}

        self.scheduler = None
        self.already_init_pipe = False
        self.already_reload_pipe = False

        self.component = {}
        self.split_controlnet = split_controlnet
        # self.init_pipe()

    @classmethod
    def init_from_other_pipe(cls, other):
        obj = cls(device=other.device, dtype=other.dtype,
                  vae_scale_factor=other.vae_scale_factor, vae_scaling_factor=other.vae_scaling_factor)

        if obj.unet_in_channels != other.unet_in_channels or obj.num_channels_latents != other.num_channels_latents:
            raise RuntimeError(
                "init_from_other_pipe won't work when unet_in_channels or num_channels_latents not the same")

        obj.unet_cache = other.unet_cache
        obj.unet_in_channels = other.unet_in_channels
        obj.controlnet_cache = other.controlnet_cache

        obj.text_encoder_cache = other.text_encoder_cache
        obj.tokenizer_cache = other.tokenizer_cache

        obj.loaded_lora = other.loaded_lora
        obj.loaded_lora_strength = other.loaded_lora_strength

        obj.scheduler = other.scheduler
        obj.split_controlnet = other.split_controlnet

        if other.already_init_pipe:
            obj.vae = other.vae
            obj.unet = other.unet
            obj.image_processor = other.image_processor
            obj.mask_processor = other.mask_processor
            obj.feature_extractor = other.feature_extractor
            obj.already_init_pipe = other.already_init_pipe

        if other.already_reload_pipe:
            obj.tokenizer = other.tokenizer
            obj.text_encoder = other.text_encoder
            obj.already_reload_pipe = other.already_reload_pipe

        return obj

    @property
    def components(self) -> Dict[str, Any]:

        cur_components = {
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder
        }

        return cur_components

    def init_pipe(self):
        if not self.already_init_pipe:
            self.vae = LyraDiffVaeModel(
                scale_factor=self.vae_scale_factor, scaling_factor=self.vae_scaling_factor)

            self.unet = LyraDiffUNet2DConditionModel(num_channels_latents=self.num_channels_latents,
                                                   num_channels_unet=self.num_channels_unet, quant_level=self.quant_level, is_sdxl=False).to(self.dtype).to(self.device)

            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor)

            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
            )

            self.feature_extractor = CLIPImageProcessor()
            self.already_init_pipe = True

    def reload_pipe(self, model_path, cache=False):

        if self.quant_level > 0:
            self.unet.load_from_bin(os.path.join(
                model_path, "unet_bins_fp16"), "fp16")
        else:
            self.unet.load_from_diffusers_model(
                os.path.join(model_path, "unet"), cache)

        # self.reload_unet_model_v2(model_path, cache)
        self.vae.load_from_diffusers_model(
            os.path.join(model_path, "vae"), cache)
        self.reload_textencoder_model_v2(model_path, cache)
        self.reload_tokenizer_v2(model_path, cache)

        if not self.scheduler:
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler", timestep_spacing="linspace")
        self.already_reload_pipe = True
        self.loaded_lora_strength = {}

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def reload_unet_model(self, unet_path, unet_file_format='fp32'):
        if len(unet_path) > 0 and unet_path[-1] != "/":
            unet_path = unet_path + "/"
        self.unet.reload_unet_model(unet_path, unet_file_format)

    def reload_vae_model(self, vae_path, vae_file_format='fp32'):
        if len(vae_path) > 0 and vae_path[-1] != "/":
            vae_path = vae_path + "/"
        return self.vae.reload_vae_model(vae_path, vae_file_format)

    def load_lora(self, lora_model_path, lora_name, lora_strength, lora_file_format='fp32'):
        if len(lora_model_path) > 0 and lora_model_path[-1] != "/":
            lora_model_path = lora_model_path + "/"
        lora = add_xltext_lora_layer(
            self.text_encoder, self.text_encoder_2, lora_model_path, lora_strength, lora_file_format)

        self.loaded_lora[lora_name] = lora
        self.unet.model.load_lora(lora_model_path, lora_name,
                                  lora_strength, lora_file_format)

    def unload_lora(self, lora_name, clean_cache=False):
        for layer_data in self.loaded_lora[lora_name]:
            layer = layer_data['layer']
            added_weight = layer_data['added_weight']
            layer.weight.data -= added_weight
        self.unet.model.unload_lora(lora_name, clean_cache)
        del self.loaded_lora[lora_name]
        gc.collect()
        torch.cuda.empty_cache()

    def load_lora_v2(self, lora_model_path, lora_name, lora_strength):
        if lora_name in self.loaded_lora:
            state_dict = self.loaded_lora[lora_name]
        else:
            state_dict = load_lora_state_dict(lora_model_path)
            self.loaded_lora[lora_name] = state_dict
        self.loaded_lora_strength[lora_name] = lora_strength
        add_lora_to_opt_model(state_dict, self.unet.model, self.text_encoder,
                              None, lora_strength)

    def unload_lora_v2(self, lora_name, clean_cache=False):
        state_dict = self.loaded_lora[lora_name]
        lora_strength = self.loaded_lora_strength[lora_name]
        add_lora_to_opt_model(state_dict, self.unet.model, self.text_encoder,
                              None,  -1.0 * lora_strength)
        del self.loaded_lora_strength[lora_name]

        if clean_cache:
            del self.loaded_lora[lora_name]
            gc.collect()
            torch.cuda.empty_cache()

    def clean_lora_cache(self):
        self.unet.clean_lora_cache()

    def get_loaded_lora(self):
        return self.unet.get_loaded_lora()

    def load_ip_adapter(self, dir_ip_adapter, ip_plus, image_encoder_path, num_ip_tokens, ip_projection_dim,  dir_face_in=None, num_fp_tokens=1, fp_projection_dim=None, sdxl=True):
        self.ip_adapter_helper = LyraIPAdapter(self.unet.model, sdxl, "cuda", dir_ip_adapter, ip_plus, image_encoder_path,
                                               num_ip_tokens, ip_projection_dim, dir_face_in, num_fp_tokens, fp_projection_dim)

    def reload_textencoder_model_v2(self, model_path, cache=False):
        if model_path in self.text_encoder_cache:
            state_dict = self.text_encoder_cache[model_path]
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=None,
                                                              config=model_path, subfolder="text_encoder", state_dict=state_dict).to(self.dtype).to(self.device)
        else:
            model = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder")
            if cache:
                self.text_encoder_cache[model_path] = model.state_dict()
            self.text_encoder = model.to(self.dtype).to(self.device)

    def reload_tokenizer_v2(self, model_path, cache=False):
        if model_path in self.tokenizer_cache:
            self.tokenzer = self.tokenizer_cache[model_path]
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer")
            if cache:
                self.tokenizer_cache[model_path] = self.tokenizer

    def load_controlnet_model_v2(self, model_name, controlnet_path, controlnet_mode="large"):
        if model_name in self.controlnet_cache:
            return
        cur_controlnet = LyraDiffControlNetModel(self.unet)
        cur_controlnet.load_from_diffusers_model(
            model_name, controlnet_path, controlnet_mode)
        self.controlnet_cache[model_name] = cur_controlnet

    def unload_controlnet_model(self, model_name):
        if model_name in self.controlnet_cache:
            del self.controlnet_cache[model_name]

    def get_loaded_controlnet(self):
        return self.unet.model.get_loaded_controlnet()

    def set_controlnet_list(self, controlnet_names):
        nets = []
        for name in controlnet_names:
            nets.append(self.controlnet_cache[name])
        self.controlnet = MultiControlNetModel(nets)
