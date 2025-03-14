import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import gc
import torch
import numpy as np
from glob import glob

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler,
                                  KarrasDiffusionSchedulers)
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from .lyradiff_vae_model import LyraDiffVaeModel
from .module.lyradiff_ip_adapter import LyraIPAdapter
from .lora_util import add_text_lora_layer, add_xltext_lora_layer, add_lora_to_opt_model, load_lora_state_dict
from safetensors.torch import load_file
from .lyradiff_unet_model import LyraDiffUNet2DConditionModel
from .lyradiff_controlnet_model import LyraDiffControlNetModel


class LyraDiffXLPipelineBase(TextualInversionLoaderMixin):
    def __init__(self, device=torch.device("cuda"), dtype=torch.float16, num_channels_unet=4, num_channels_latents=4, vae_scale_factor=8, vae_scaling_factor=0.13025, split_controlnet=False, quant_level=0, use_fp16_vae=False) -> None:
        self.device = device
        self.dtype = dtype

        self.quant_level = quant_level

        self.num_channels_unet = num_channels_unet
        self.num_channels_latents = num_channels_latents
        self.vae_scale_factor = vae_scale_factor
        self.vae_scaling_factor = vae_scaling_factor

        self.unet_cache = {}
        self.unet_in_channels = 4

        self.text_encoder_cache = {'text_encoder': {}, 'text_encoder_2': {}}
        self.tokenizer_cache = {'tokenizer': {}, 'tokenizer_2': {}}

        self.controlnet_cache = {}
        self.controlnet_add_embedding = {}

        self.loaded_lora = {}
        self.loaded_lora_strength = {}

        self.scheduler = None

        self.already_init_pipe = False
        self.already_reload_pipe = False
        self.split_controlnet = split_controlnet

        self.use_fp16_vae = use_fp16_vae
        # self.init_pipe()

    @property
    def components(self) -> Dict[str, Any]:

        cur_components = {
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "tokenizer_2": self.tokenizer_2,
            "text_encoder_2": self.text_encoder_2
        }

        return cur_components

    @classmethod
    def init_from_other_pipe(cls, other):
        obj = cls(device=other.device, dtype=other.dtype,
                  vae_scale_factor=other.vae_scale_factor, vae_scaling_factor=other.vae_scaling_factor)

        if obj.unet_in_channels != other.unet_in_channels or obj.num_channels_latents != other.num_channels_latents:
            raise RuntimeError(
                "init_from_other_pipe won't work when unet_in_channels or num_channels_latents not the same")

        obj.unet_cache = other.unet_cache
        obj.controlnet_cache = other.controlnet_cache
        obj.controlnet_add_embedding = other.controlnet_add_embedding

        obj.text_encoder_cache = other.text_encoder_cache
        obj.tokenizer_cache = other.tokenizer_cache

        obj.loaded_lora = other.loaded_lora
        obj.loaded_lora_strength = other.loaded_lora_strength

        obj.scheduler = other.scheduler

        obj.split_controlnet = other.split_controlnet

        if other.already_init_pipe:
            obj.already_init_pipe = other.already_init_pipe

            obj.vae = other.vae
            obj.unet = other.unet
            obj.default_sample_size = other.default_sample_size
            obj.addition_time_embed_dim = other.default_sample_size
            obj.flip_sin_to_cos, obj.freq_shift = True, 0
            obj.projection_class_embeddings_input_dim, obj.time_embed_dim = 2816, 1280

            obj.add_time_proj = other.add_time_proj
            obj.add_embedding = other.add_embedding
            obj.image_processor = other.image_processor
            obj.mask_processor = other.mask_processor
            obj.control_image_processor = other.control_image_processor
            obj.feature_extractor = other.feature_extractor

        if other.already_reload_pipe:
            obj.tokenizer = other.tokenizer
            obj.text_encoder = other.text_encoder
            obj.tokenizer_2 = other.tokenizer_2
            obj.text_encoder_2 = other.text_encoder_2

            obj.already_reload_pipe = other.already_reload_pipe

        return obj

    def init_pipe(self):
        if not self.already_init_pipe:
            self.vae = LyraDiffVaeModel(
                scale_factor=self.vae_scale_factor, scaling_factor=self.vae_scaling_factor, is_upcast=(not self.use_fp16_vae))

            # self.unet = torch.classes.lyradiff.XLUnet2dConditionalModelOp(
            #     "fp16",
            #     self.num_channels_unet,
            #     self.num_channels_latents,
            #     self.quant_level)

            self.unet = LyraDiffUNet2DConditionModel(num_channels_latents=self.num_channels_latents,
                                                   num_channels_unet=self.num_channels_unet, quant_level=self.quant_level, is_sdxl=True).to(self.dtype).to(self.device)

            self.default_sample_size = 128
            self.addition_time_embed_dim = 256
            flip_sin_to_cos, freq_shift = True, 0
            self.projection_class_embeddings_input_dim, self.time_embed_dim = 2816, 1280

            self.add_time_proj = Timesteps(
                self.addition_time_embed_dim, flip_sin_to_cos, freq_shift).to(self.dtype).to(self.device)

            self.add_embedding = TimestepEmbedding(
                self.projection_class_embeddings_input_dim, self.time_embed_dim).to(self.dtype).to(self.device)

            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor)

            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
            )

            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
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

        if self.use_fp16_vae:
            print("Since we are in use fp16 vae mode, please load vae sepreately")
        else:
            self.vae.load_from_diffusers_model(
                os.path.join(model_path, "vae"), cache)

        self.reload_textencoder_model_v2(model_path, cache)
        self.reload_tokenizer_v2(model_path, cache)

        if not self.scheduler:
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler", timestep_spacing="linspace")
        self.already_reload_pipe = True

    def load_embedding_weight(self, model, weight_path, unet_file_format="fp16"):
        bin_list = glob(weight_path)
        sate_dicts = model.state_dict()
        dtype = np.float32 if unet_file_format == "fp32" else np.float16
        for bin_file in bin_list:
            weight = torch.from_numpy(np.fromfile(bin_file, dtype=dtype)).to(
                self.dtype).to(self.device)
            key = '.'.join(os.path.basename(bin_file).split('.')[1:-1])
            weight = weight.reshape(sate_dicts[key].shape)
            sate_dicts.update({key: weight})
        model.load_state_dict(sate_dicts)

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
        self.unet.load_from_bin(unet_path, unet_file_format)

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

    def load_lora_v2(self, lora_model_path, lora_name, lora_strength, need_trans=True):
        if lora_name in self.loaded_lora:
            state_dict = self.loaded_lora[lora_name]
        else:
            state_dict = load_lora_state_dict(lora_model_path, need_trans)
            self.loaded_lora[lora_name] = state_dict
        self.loaded_lora_strength[lora_name] = lora_strength
        add_lora_to_opt_model(state_dict, self.unet.model, self.text_encoder,
                              self.text_encoder_2, lora_strength)

    def unload_lora_v2(self, lora_name, clean_cache=False):
        state_dict = self.loaded_lora[lora_name]
        lora_strength = self.loaded_lora_strength[lora_name]
        add_lora_to_opt_model(state_dict, self.unet.model, self.text_encoder,
                              self.text_encoder_2,  -1.0 * lora_strength)
        del self.loaded_lora_strength[lora_name]

        if clean_cache:
            del self.loaded_lora[lora_name]
            gc.collect()
            torch.cuda.empty_cache()

    def clean_lora_cache(self):
        self.unet.clean_lora_cache()

    def get_loaded_lora(self):
        return self.unet.get_loaded_lora()

    def _get_aug_emb(self, time_ids, text_embeds, dtype):
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(dtype)
        aug_emb = self.add_embedding(add_embeds)
        return aug_emb

    def load_ip_adapter(self, dir_ip_adapter, ip_plus, image_encoder_path, num_ip_tokens, ip_projection_dim,  dir_face_in=None, num_fp_tokens=1, fp_projection_dim=None, sdxl=True):
        self.ip_adapter_helper = LyraIPAdapter(self.unet.model, sdxl, "cuda", dir_ip_adapter, ip_plus, image_encoder_path,
                                               num_ip_tokens, ip_projection_dim, dir_face_in, num_fp_tokens, fp_projection_dim)

    def reload_textencoder_model_v2(self, model_path, cache=False):
        if model_path in self.text_encoder_cache['text_encoder']:
            state_dict = self.text_encoder_cache['text_encoder'][model_path]
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=None,
                                                              config=model_path, subfolder="text_encoder", state_dict=state_dict).to(self.dtype).to(self.device)

            state_dict2 = self.text_encoder_cache['text_encoder_2'][model_path]
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=None,
                                                                            config=model_path, subfolder="text_encoder_2", state_dict=state_dict2).to(self.dtype).to(self.device)
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder")
            self.text_encoder = text_encoder.to(self.dtype).to(self.device)

            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model_path, subfolder="text_encoder_2")
            self.text_encoder_2 = text_encoder_2.to(self.dtype).to(self.device)

            if cache:
                self.text_encoder_cache['text_encoder'][model_path] = text_encoder.state_dict(
                )
                self.text_encoder_cache['text_encoder_2'][model_path] = text_encoder_2.state_dict(
                )

    def reload_tokenizer_v2(self, model_path, cache=False):
        if model_path in self.tokenizer_cache['tokenizer']:
            self.tokenzer = self.tokenizer_cache['tokenizer'][model_path]
            self.tokenzer_2 = self.tokenizer_cache['tokenizer_2'][model_path]
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2")
            if cache:
                self.tokenizer_cache['tokenizer'][model_path] = self.tokenizer
                self.tokenizer_cache['tokenizer_2'][model_path] = self.tokenizer_2

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