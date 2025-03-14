import os, sys
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.models.embeddings import ImageProjection
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
import time
sys.path.append(os.path.dirname(__file__))
from resampler import Resampler
from diffusers import DiffusionPipeline
import numpy as np
# sys.path.append(os.environ['LYRADIFF_WORKDIR'] + "/tests/utils")
from .tools import get_mem_use

class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class LyraIPAdapter:
    def __init__(
            self,
            unet_model,
            sdxl,
            device,
            ip_ckpt=None,
            ip_plus=False,
            image_encoder_path=None,
            num_ip_tokens=4,
            ip_projection_dim=None,
            cross_attention_dim=2048
        ):
        self.unet_model = unet_model
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_ip_tokens = num_ip_tokens
        self.ip_projection_dim = 1024
        self.sdxl = sdxl
        self.ip_plus = ip_plus
        self.cross_attention_dim = cross_attention_dim

        if self.ip_plus:
            self.num_ip_tokens = 16
            self.cross_attention_dim = 2048
        else:
            self.num_ip_tokens = 4
            self.cross_attention_dim = 768

        if image_encoder_path:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=torch.float16)
            self.clip_image_processor = CLIPImageProcessor()
            self.projection_dim = self.image_encoder.config.projection_dim

        # image proj model
        if self.ip_ckpt:
            if self.ip_plus:
                proj_heads = 20 if self.sdxl else 12
                self.image_proj_model = self.init_proj_plus(proj_heads, self.num_ip_tokens)
            else:
                self.image_proj_model = self.init_proj(self.ip_projection_dim, self.num_ip_tokens)

        self.load_ip_adapter()
        
    def init_proj_diffuser(self, state_dict):
        # diffusers加载版本
        clip_embeddings_dim = state_dict["image_proj"]["proj.weight"].shape[-1]
        cross_attention_dim = state_dict["image_proj"]["proj.weight"].shape[0] // 4

        image_proj_model = ImageProjection(
            cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim, num_image_text_embeds=4
        ).to(dtype=self.dtype, device=self.device)
        return image_proj_model

    # init_proj / init_proj_plus 
    def init_proj(self, projection_dim, num_tokens):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=projection_dim,
            clip_extra_context_tokens=num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


    def init_proj_plus(self, heads, num_tokens):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=heads,
            num_queries=num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
        
    def load_ip_adapter(self):

        def parse_ckpt_path(ckpt):
            ll = ckpt.split("/")
            weight_name = ll[-1]
            subfolder = ll[-2]
            pretrained_path = "/".join(ll[:-2])
            return pretrained_path, subfolder, weight_name

        if self.ip_ckpt:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            pretrained_path, subfolder, weight_name = parse_ckpt_path(self.ip_ckpt)
            dir_ipadapter = os.path.join(pretrained_path, subfolder, '.'.join(weight_name.split(".")[:-1]))
            self.unet_model.load_ip_adapter(dir_ipadapter, "", 1, "fp16")
        
    @torch.inference_mode()
    def get_image_embeds(self, image=None):
        image_prompt_embeds, uncond_image_prompt_embeds = None, None

        if image is not None:
            if not isinstance(image, list):
                image = [image]
            clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            if self.ip_plus:
                clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
                uncond_clip_image_embeds = self.image_encoder(
                    torch.zeros_like(clip_image), output_hidden_states=True
                ).hidden_states[-2]
            else:
                clip_image_embeds = self.image_encoder(clip_image).image_embeds
                uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)
            clip_image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            image_prompt_embeds = clip_image_prompt_embeds
            uncond_image_prompt_embeds = uncond_clip_image_prompt_embeds

        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_image_embeds_lyradiff(self, image=None, ip_image_embeds=None, batch_size = 1, ip_scale=1.0, do_classifier_free_guidance=True):
        dict_tensor = {}

        if self.ip_ckpt and ip_scale>0:
            if ip_image_embeds is not None:
                dict_tensor["ip_hidden_states"] = ip_image_embeds
            elif image is not None:
                if not isinstance(image, list):
                    image = [image]
                clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
                clip_image = clip_image.to(self.device, dtype=torch.float16)
                if self.ip_plus:
                    clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
                    uncond_clip_image_embeds = self.image_encoder(
                        torch.zeros_like(clip_image), output_hidden_states=True
                    ).hidden_states[-2]
                else:
                    clip_image_embeds = self.image_encoder(clip_image).image_embeds
                    uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

                if do_classifier_free_guidance:
                    clip_image_embeds = torch.cat([uncond_clip_image_embeds, clip_image_embeds])
                ip_image_embeds = self.image_proj_model(clip_image_embeds)
                dict_tensor["ip_hidden_states"] = ip_image_embeds

        return dict_tensor

    