import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.controlnet import ControlNetOutput, ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import logging
from .lyradiff_unet_model import LyraDiffUNet2DConditionModel
from safetensors.torch import load_file
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .lyradiff_utils import get_aug_emb, load_embedding_weight
from diffusers.configuration_utils import FrozenDict

logger = logging.get_logger(__name__)


class LyraDiffControlNetModel(ControlNetModel):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, lyra_unet_model: LyraDiffUNet2DConditionModel):
        super().__init__()

        self._dtype = lyra_unet_model.dtype
        self._device = lyra_unet_model.device

        self.controlnet_name = ""
        self.model = lyra_unet_model.model
        self.add_embedding_cache = None
        # self.controlnet_cache = {}
        # self.aug_cache = {}

        self.is_sdxl = lyra_unet_model.is_sdxl
        self.output_len = 10 if self.is_sdxl else 13

        self.prev_t = 0

        self.default_sample_size = 128
        self.addition_time_embed_dim = 256
        flip_sin_to_cos, freq_shift = True, 0
        self.projection_class_embeddings_input_dim, self.time_embed_dim = 2816, 1280

        self.add_time_proj = Timesteps(
            self.addition_time_embed_dim, flip_sin_to_cos, freq_shift).to(self._dtype).to(self._device)

        self.add_embedding = TimestepEmbedding(
            self.projection_class_embeddings_input_dim, self.time_embed_dim).to(self._dtype).to(self._device)

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(
                f"Make sure that {self.__class__} has defined a class name `config_name`")
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

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(
            self.__dict__["_internal_dict"], name)
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

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

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
            config, kwargs = ControlNetModel.load_config(
                pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, _, hidden_dict = ControlNetModel.extract_init_dict(
            config, **kwargs)
        if "_class_name" in init_dict:
            init_dict["_class_name"] = ControlNetModel.__name__
        self.config_name = "config.json"
        self.register_to_config(**init_dict)

    def __del__(self):
        self.unload_controlnet_model(self.controlnet_name)

    def convert_state_dict(self, state_dict):
        for key in state_dict:
            if len(state_dict[key].shape) == 4:
                state_dict[key] = state_dict[key].to(
                    torch.float16).permute(0, 2, 3, 1).contiguous()
            state_dict[key] = state_dict[key].to(torch.float16)

        return state_dict

    def load_from_diffusers_model(self, model_name, controlnet_path, controlnet_mode="large"):
        config_path = os.path.join(
            controlnet_path, "config.json")
        
        self.load_config(config_path)
        
        checkpoint_file = os.path.join(
            controlnet_path, "diffusion_pytorch_model.bin")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(
                controlnet_path, "diffusion_pytorch_model.safetensors")

        if "safetensors" in checkpoint_file:
            state_dict = load_file(checkpoint_file)
        else:
            state_dict = torch.load(checkpoint_file, map_location="cpu")

        state_dict = self.convert_state_dict(state_dict)

        self.load_state_dict(state_dict, model_name, controlnet_mode)

    # 从state dict load 参数，需要转换过才可以
    def load_state_dict(self, state_dict, model_name, controlnet_mode="large"):
        if self.is_sdxl:
            self.model.load_controlnet_model_from_state_dict(
                model_name, state_dict, "cpu", controlnet_mode)

            load_embedding_weight(self.add_embedding, state_dict)
        else:
            self.model.load_controlnet_model_from_state_dict(
                model_name, state_dict, "cpu")
        self.controlnet_name = model_name

    def unload_controlnet_model(self, model_name):
        self.model.unload_controlnet_model(model_name, True)

    def get_loaded_controlnet(self):
        return self.model.get_loaded_controlnet()

    def get_aug(self, added_cond_kwargs, is_first_step):
        if is_first_step:
            # 只有第一步unet需要处理add_time_ids
            add_time_ids, add_text_embeds = added_cond_kwargs[
                "time_ids"], added_cond_kwargs["text_embeds"]

            aug_emb = get_aug_emb(self.add_time_proj, self.add_embedding,
                                  add_time_ids, add_text_embeds, self._dtype)
            self.add_embedding_cache = aug_emb
        else:
            aug_emb = self.add_embedding_cache
        return aug_emb

    def get_controlnet_scale(self, scale, guess_mode):
        scales = [1.0, ] * self.output_len
        if guess_mode:
            scales = torch.logspace(-1, 0, self.output_len).tolist()

        return [d * scale for d in scales]

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:

        sample = sample.permute(0, 2, 3, 1).contiguous()

        # 处理kv cache
        is_first_step = self.prev_t < timestep
        self.prev_t = timestep
        # 判断是否第一步
        if is_first_step:
            os.environ["LyraDiff_KV_CACHE_FIRST_STEP"] = "1"
        else:
            os.environ["LyraDiff_KV_CACHE_FIRST_STEP"] = "0"

        if controlnet_cond.shape[1] == 3:
            controlnet_cond = controlnet_cond[0].unsqueeze(0).permute(
                0, 2, 3, 1).contiguous()

        if self.is_sdxl:
            aug = self.get_aug(added_cond_kwargs, is_first_step)
            scale = self.get_controlnet_scale(conditioning_scale, guess_mode)
            cur_controlnet_results = self.model.controlnet_forward(
                sample, encoder_hidden_states, controlnet_cond, aug, timestep, scale, self.controlnet_name)
        else:
            scale = self.get_controlnet_scale(conditioning_scale, guess_mode)
            cur_controlnet_results = self.model.controlnet_forward(
                sample, encoder_hidden_states, controlnet_cond, timestep, scale, self.controlnet_name)

        down_samples, mid_sample = cur_controlnet_results[:-1], cur_controlnet_results[-1]
        if not return_dict:
            return (down_samples, mid_sample)

        return ControlNetOutput(
            down_block_res_samples=down_samples, mid_block_res_sample=mid_sample
        )
