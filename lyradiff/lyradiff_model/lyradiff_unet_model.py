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


import diffusers
from diffusers import UNet2DConditionModel
if hasattr(diffusers.models, "unet_2d_condition"):
    from diffusers.models.unet_2d_condition import UNet2DConditionOutput
else:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
# from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict

from .lyradiff_utils import get_aug_emb, load_embedding_weight, LyraQuantLevel


class LyraDiffUNet2DConditionModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float16, num_channels_unet=4, num_channels_latents=4, quant_level=LyraQuantLevel.NONE, is_sdxl=False
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.quant_level = quant_level.value

        self.num_channels_unet = num_channels_unet
        self.num_channels_latents = num_channels_latents

        self.cache = {}
        self.unet_in_channels = 4

        self.is_sdxl = is_sdxl

        if self.is_sdxl:
            self.model = torch.classes.lyradiff.XLUnet2dConditionalModelOp(
                "fp16",
                self.num_channels_unet,
                self.num_channels_latents,
                self.quant_level)

            self.default_sample_size = 128
            self.addition_time_embed_dim = 256
            flip_sin_to_cos, freq_shift = True, 0
            self.projection_class_embeddings_input_dim, self.time_embed_dim = 2816, 1280

            self.add_time_proj = Timesteps(
                self.addition_time_embed_dim, flip_sin_to_cos, freq_shift).to(self.dtype).to(self.device)

            self.add_embedding = TimestepEmbedding(
                self.projection_class_embeddings_input_dim, self.time_embed_dim).to(self.dtype).to(self.device)
        else:
            self.model = torch.classes.lyradiff.Unet2dConditionalModelOp(
                3,
                "fp16",
                self.num_channels_unet,
                self.num_channels_latents,
                self.quant_level
            )

        self.prev_t = 0

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
            config, kwargs = UNet2DConditionModel.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, _, hidden_dict = UNet2DConditionModel.extract_init_dict(config, **kwargs)
        if "_class_name" in init_dict:
            init_dict["_class_name"] = UNet2DConditionModel.__name__
        self.config_name = "config.json"
        self.register_to_config(**init_dict)

    def convert_state_dict(self, state_dict):
        for key in state_dict:
            if len(state_dict[key].shape) == 4:
                state_dict[key] = state_dict[key].to(
                    torch.float16).permute(0, 2, 3, 1).contiguous()
            state_dict[key] = state_dict[key].to(torch.float16)

        return state_dict

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

    def load_from_bin(self, unet_path, unet_file_format='fp32'):
        if len(unet_path) > 0 and unet_path[-1] != "/":
            unet_path = unet_path + "/"
        self.model.reload_unet_model(unet_path, unet_file_format)

        if self.is_sdxl:
            self.load_embedding_weight(
                self.add_embedding, f"{unet_path}add_embedding*", unet_file_format=unet_file_format)

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

            if cache:
                self.cache[checkpoint_file] = state_dict

        self.load_state_dict(state_dict)

    # 从state dict load 参数，需要转换过才可以
    def load_state_dict(self, state_dict):
        self.model.reload_unet_model_from_cache(state_dict, "cpu")

        if self.is_sdxl:
            load_embedding_weight(self.add_embedding, state_dict)

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale

    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        # 因为我们的推理为NHWC，所以需要对输入进行permute
        sample = sample.permute(0, 2, 3, 1).contiguous()

        # 处理Controlnet输入数据
        controlnet_input = []
        if down_block_additional_residuals is not None:
            for c in down_block_additional_residuals:
                controlnet_input.append(c.contiguous())
            controlnet_input.append(mid_block_additional_residual.contiguous())

        # 处理Ip Adapter 输入
        param_scale_dict = {}
        extra_tensor_dict = {}

        if added_cond_kwargs is not None and "image_embeds" in added_cond_kwargs:
            param_scale_dict["ip_ratio"] = self.ip_adapter_scale
            extra_tensor_dict["ip_hidden_states"] = added_cond_kwargs["image_embeds"][0]

        # 处理kv cache
        is_first_step = self.prev_t < timestep
        self.prev_t = timestep
        # 判断是否第一步
        if is_first_step:
            os.environ["LYRADIFF_KV_CACHE_FIRST_STEP"] = "1"
        else:
            os.environ["LYRADIFF_KV_CACHE_FIRST_STEP"] = "0"

        if self.is_sdxl:
            if is_first_step:
                # 只有第一步unet需要处理add_time_ids
                add_time_ids, add_text_embeds = added_cond_kwargs[
                    "time_ids"], added_cond_kwargs["text_embeds"]

                aug_emb = get_aug_emb(self.add_time_proj, self.add_embedding,
                                      add_time_ids, add_text_embeds, encoder_hidden_states.dtype)
                self.aug_emb = aug_emb
            else:
                aug_emb = self.aug_emb

            sample = self.model.unet_forward(
                sample, encoder_hidden_states, aug_emb, timestep, controlnet_input, extra_tensor_dict, param_scale_dict).permute(0, 3, 1, 2)
        else:
            sample = self.model.unet_forward(
                sample, encoder_hidden_states, timestep, controlnet_input, extra_tensor_dict, param_scale_dict).permute(0, 3, 1, 2)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
