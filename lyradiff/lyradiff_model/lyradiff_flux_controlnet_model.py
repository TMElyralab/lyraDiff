import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.controlnet_flux import FluxControlNetOutput, FluxControlNetModel
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.utils import logging
from safetensors.torch import load_file
from diffusers.configuration_utils import FrozenDict
from .lyradiff_utils import get_lyradiff_context

logger = logging.get_logger(__name__)

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
    
def load_hint_block_weight(model, state_dict):
    print("loading load_hint_block_weight")
    sub_state_dict = {}
    for k in state_dict:
        if k.startswith("input_hint_block"):
            v = state_dict[k]
            sub_k = ".".join(k.split(".")[1:])
            sub_state_dict[sub_k] = v

    model.load_state_dict(sub_state_dict)

class LyraDiffFluxControlNetModel(FluxControlNetModel):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, 
        device=torch.device("cuda"),
        dtype=torch.bfloat16, 
        in_channels: int = 64,
        num_layers: int = 2,
        num_single_layers: int = 0,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        conditioning_embedding_channels: int = 16,
        quant_level = 0):

        super().__init__(num_layers=0, num_single_layers=0)
        self.device_ = device
        self.dtype_ = dtype

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds

        self.quant_level = quant_level

        self.model = torch.classes.lyradiff.FluxControlnetModelOp(
            "bf16",
            self.in_channels,
            self.num_layers,
            self.num_single_layers,
            self.attention_head_dim,
            self.num_attention_heads,
            self.pooled_projection_dim,
            self.joint_attention_dim,
            self.guidance_embeds,
            self.quant_level)
        self.model.initialize(get_lyradiff_context())
        self.inner_dim = attention_head_dim * num_attention_heads

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=axes_dims_rope)

        self.input_hint_block = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=conditioning_embedding_channels, block_out_channels=(16, 16, 16, 16)
        ).to(self.dtype_).to(self.device_)

        self.need_input_hint_block = False

        self.prev_t = 0

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
            config, kwargs = FluxControlNetModel.load_config(
                pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, _, hidden_dict = FluxControlNetModel.extract_init_dict(
            config, **kwargs)
        if "_class_name" in init_dict:
            init_dict["_class_name"] = FluxControlNetModel.__name__
        self.config_name = "config.json"
        self.register_to_config(**init_dict)

    def convert_state_dict(self, state_dict):
        for key in state_dict:
            state_dict[key] = state_dict[key].to(self.dtype_)
        return state_dict

    def load_from_diffusers_model(self, controlnet_path):
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
        num_layers = self.config.num_layers
        num_single_layers = self.config.num_single_layers
        self.load_state_dict(state_dict, num_layers, num_single_layers)
        if "input_hint_block.blocks.0.weight" in state_dict:
            self.need_input_hint_block = True
            load_hint_block_weight(self.input_hint_block, state_dict)
        else:
            self.need_input_hint_block = False

    # 从state dict load 参数，需要转换过才可以
    def load_state_dict(self, state_dict, num_layers, num_single_layers):
        self.model.reload_controlnet_model_from_state_dict(state_dict, num_layers, num_single_layers, "cpu")

    def preprocess_controlnet_cond(self, controlnet_cond):
        if not self.need_input_hint_block:
            return controlnet_cond
        controlnet_cond = self.input_hint_block(controlnet_cond)
        batch_size, channels, height_pw, width_pw = controlnet_cond.shape
        height = height_pw // self.config.patch_size
        width = width_pw // self.config.patch_size
        controlnet_cond = controlnet_cond.reshape(
            batch_size, channels, height, self.config.patch_size, width, self.config.patch_size
        )
        print("preprocess_controlnet_cond")
        controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
        controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1).contiguous()
        return controlnet_cond

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        timestep = (timestep.to(self.dtype_) * 1000)[0].item()
        guidance = (guidance.to(self.dtype_) * 1000)[0].item()

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
            image_rotary_emb = self.pos_embed(ids).to(self.dtype_)
            self.image_rotary_emb = image_rotary_emb

            self.controlnet_cond = self.preprocess_controlnet_cond(controlnet_cond)
            
        else:
            os.environ["LyraDiff_KV_CACHE_FIRST_STEP"] = "0"

        # print("LyraDiff processed controlnet_cond: ", self.controlnet_cond)

        res = self.model.controlnet_forward(
            hidden_states, encoder_hidden_states, self.image_rotary_emb, pooled_projections, self.controlnet_cond, timestep, guidance, conditioning_scale)
        
        # print(len(res))
        controlnet_block_samples, controlnet_single_block_samples = res[:self.config.num_layers], res[self.config.num_layers:]
        # print(len(controlnet_block_samples))
        # print(len(controlnet_single_block_samples))

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
