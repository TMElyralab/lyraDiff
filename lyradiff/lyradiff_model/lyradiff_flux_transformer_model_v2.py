# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple, Union
from glob import glob
import torch
import torch.nn as nn

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import numpy as np

from safetensors.torch import load_file
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

import os

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.normalization import AdaLayerNormContinuous

from .lyradiff_utils import get_lyradiff_context, LyraQuantLevel

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64,
                         device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta)
             for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

class LyraDiffFluxTransformerModel(nn.Module):
    def __init__(
        self, 
        device=torch.device("cuda"),
        dtype=torch.bfloat16, 
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        quant_level = LyraQuantLevel.NONE
    ):
        super().__init__()
        self.model = torch.classes.lyradiff.FluxTransformerBlockOp(
            "bf16",
            attention_head_dim*num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            6,
            quant_level.value
        )

        self.model.initialize(get_lyradiff_context())

    def load_from_bin(self, transformer_path, index, transformer_file_format='fp16'):
        if len(transformer_path) > 0 and transformer_path[-1] != "/":
            transformer_path = transformer_path + "/"
        transformer_path = transformer_path + f"transformer_blocks.{index}."
        self.model.reload_model_from_bin(transformer_path, transformer_file_format)

    def load_state_dict(self, state_dict, index):
        prefix = f"transformer_blocks.{index}."
        self.model.reload_model_from_state_dict(prefix, state_dict, "cpu")

    def load_lora(self, layers, curr_layer_weight, lora_scale):
        self.model.load_lora_by_name(layers, curr_layer_weight, lora_scale)

    def clear_lora(self):
        self.model.clear_lora()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        # res[1] is encoder res and res[0] is hidden res
        res = self.model.forward(hidden_states, encoder_hidden_states, image_rotary_emb, temb)

        return res[1], res[0]

class LyraDiffFluxSingleTransformerModel(nn.Module):
    def __init__(
        self, 
        device=torch.device("cuda"),
        dtype=torch.bfloat16, 
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        quant_level = LyraQuantLevel.NONE
    ):
        super().__init__()
        self.model = torch.classes.lyradiff.FluxSingleTransformerBlockOp(
            "bf16",
            attention_head_dim*num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            4,
            quant_level.value
        )

        self.model.initialize(get_lyradiff_context())

    def load_from_bin(self, transformer_path, index, transformer_file_format='fp16'):
        if len(transformer_path) > 0 and transformer_path[-1] != "/":
            transformer_path = transformer_path + "/"
        transformer_path = transformer_path + f"single_transformer_blocks.{index}."
        self.model.reload_model_from_bin(transformer_path, transformer_file_format)

    def load_state_dict(self, state_dict, index):
        prefix = f"single_transformer_blocks.{index}."
        self.model.reload_model_from_state_dict(prefix, state_dict, "cpu")

    def load_lora(self, layers, curr_layer_weight, lora_scale):
        self.model.load_lora_by_name(layers, curr_layer_weight, lora_scale)

    def clear_lora(self):
        self.model.clear_lora()

    def forward(self, 
                hidden_states: torch.FloatTensor,
                temb: torch.FloatTensor,
                image_rotary_emb=None,
                joint_attention_kwargs=None,):
        return self.model.forward(hidden_states, image_rotary_emb, temb)


class LyraDiffFluxTransformer2DModelV2(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self, 
        device=torch.device("cuda"),
        dtype=torch.bfloat16, 
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        quant_level = LyraQuantLevel.NONE
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds

        self.quant_level = quant_level
        
        # self.model.initialize(get_lyradiff_context())
        self.inner_dim = attention_head_dim * num_attention_heads

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=axes_dims_rope)

        self.cache = {}

        self.prev_t = 0

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.pooled_projection_dim
        ).to(dtype=dtype, device=device)

        self.context_embedder = nn.Linear(self.joint_attention_dim, self.inner_dim, dtype=dtype, device=device)
        self.x_embedder = nn.Linear(self.in_channels, self.inner_dim, dtype=dtype, device=device)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6).to(dtype=dtype, device=device)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [
                LyraDiffFluxTransformerModel(
                    device=device,
                    dtype=dtype, 
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
                    quant_level = quant_level
                )
                for _ in range(self.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                LyraDiffFluxSingleTransformerModel(
                    device=device,
                    dtype=dtype, 
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
                    quant_level = quant_level
                )
                for _ in range(self.num_single_layers)
            ]
        )

        self.silu = torch.nn.SiLU()

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}

        self._internal_dict = FrozenDict(internal_dict)

    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129

        This function is mostly copied from PyTorch's __getattr__ overwrite:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
            return self._internal_dict[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict

    def load_config(self, config, **kwargs):
        if not isinstance(config, dict):            
            config, kwargs = FluxTransformer2DModel.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, _, hidden_dict = FluxTransformer2DModel.extract_init_dict(config, **kwargs)
        if "_class_name" in init_dict:
            init_dict["_class_name"] = FluxTransformer2DModel.__name__
        self.config_name = "config.json"
        self.register_to_config(**init_dict)

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

    # 目前适用于量化/转换过的模型
    def load_from_bin(self, transformer_path, transformer_file_format='fp16'):
        if len(transformer_path) > 0 and transformer_path[-1] != "/":
            transformer_path = transformer_path + "/"
        # self.model.reload_transformer_model(transformer_path, transformer_file_format)

        self.load_embedding_weight(self.time_text_embed, f"{transformer_path}time_text_embed*", unet_file_format=transformer_file_format)
        self.load_embedding_weight(self.context_embedder, f"{transformer_path}context_embedder*", unet_file_format=transformer_file_format)
        self.load_embedding_weight(self.x_embedder, f"{transformer_path}x_embedder*", unet_file_format=transformer_file_format)
        self.load_embedding_weight(self.norm_out, f"{transformer_path}norm_out*", unet_file_format=transformer_file_format)
        self.load_embedding_weight(self.proj_out, f"{transformer_path}proj_out*", unet_file_format=transformer_file_format)

        for i, block in enumerate(self.single_transformer_blocks):
            block.load_from_bin(transformer_path, i)

        for i, block in enumerate(self.transformer_blocks):
            block.load_from_bin(transformer_path, i)

    def clear_lora(self):
        self.model.clear_lora()


    def convert_state_dict(self, state_dict):
        for key in state_dict:
            if "input_scale" in key:
                state_dict[key] = state_dict[key].to(torch.float)
            else:
                state_dict[key] = state_dict[key].to(self.dtype)
        return state_dict

    # 适用于直接从diffusers load模型
    def load_from_diffusers_model(self, model_path, cache=False):
        # 寻找模型文件
        config_path = os.path.join(
            model_path, "config.json")
        
        self.load_config(config_path)

        checkpoint_file = os.path.join(
            model_path, "diffusion_pytorch_model.bin")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(
                model_path, "diffusion_pytorch_model.safetensors")

        if checkpoint_file in self.cache:
            state_dict = self.cache[checkpoint_file]
        else:
            if "safetensors" in checkpoint_file:
                state_dict = load_file(checkpoint_file, device="cpu")
            else:
                state_dict = torch.load(checkpoint_file, map_location="cpu")

            state_dict = self.convert_state_dict(state_dict)

        self.load_state_dict(state_dict)

    # 从state dict load 参数，需要转换过才可以
    def load_state_dict(self, state_dict):
        # super().load_state_dict(state_dict)
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        super().load_state_dict(
            state_dict,
            False
        )
        
        for i, block in enumerate(self.single_transformer_blocks):
            block.load_state_dict(state_dict, i)

        for i, block in enumerate(self.transformer_blocks):
            block.load_state_dict(state_dict, i)
        # pass
        # self.model.reload_transformer_model_from_state_dict(state_dict, "cpu")

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        # timestep = (timestep.to(self.dtype) * 1000)[0].item()
        # guidance = (guidance.to(self.dtype) * 1000)[0].item()

        # 处理kv cache
        is_first_step = self.prev_t < timestep
        self.prev_t = timestep
        # 判断是否第一步
        if is_first_step:
            os.environ["LyraDiff_KV_CACHE_FIRST_STEP"] = "1"

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            ids = ids.unsqueeze(0)
            image_rotary_emb = self.pos_embed(ids).to(self.dtype)
            self.image_rotary_emb = image_rotary_emb
        else:
            os.environ["LyraDiff_KV_CACHE_FIRST_STEP"] = "0"

        hidden_states = self.x_embedder(hidden_states).contiguous()

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states).contiguous()

        silu_temb = self.silu(temb).contiguous()
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states.contiguous(),
                encoder_hidden_states=encoder_hidden_states.contiguous(),
                temb=silu_temb,
                image_rotary_emb=self.image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states.contiguous(),
                temb=silu_temb,
                image_rotary_emb=self.image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
 
