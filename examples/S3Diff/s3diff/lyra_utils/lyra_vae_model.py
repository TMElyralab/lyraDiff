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

import torch
import torch.nn as nn

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import numpy as np

from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.configuration_utils import FrozenDict

from safetensors.torch import load_file

import os
import math

def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]


class LyraDiffVaeModel():
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        dtype: str = torch.float16,
        scaling_factor: float = 0.18215,
        scale_factor: int = 8,
        is_upcast: bool = False,
        device=torch.device("cuda"),

    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.is_upcast = is_upcast
        self.scaling_factor = scaling_factor
        self.scale_factor = scale_factor
        self.model = torch.classes.lyradiff.VaeModelOp(
            "fp16",
            is_upcast
        )

        self.vae_cache = {}

        self.use_slicing = False
        self.use_tiling = False

        self.tile_latent_min_size = 512
        self.tile_sample_min_size = 64
        self.tile_overlap_factor = 0.25

        self.is_dynamic_tiling_s3diff = False

        self.encode_tile_size = 1024
        self.encode_pad = 32
        self.decode_tile_size = 224
        self.decode_pad = 11

    
    def enable_dynamic_tiling_s3diff(self):
        self.is_dynamic_tiling_s3diff = True

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
            config, kwargs = AutoencoderKL.load_config(
                pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, _, hidden_dict = AutoencoderKL.extract_init_dict(
            config, **kwargs)
        if "_class_name" in init_dict:
            init_dict["_class_name"] = AutoencoderKL.__name__

        self.config_name = "config.json"
        self.register_to_config(**init_dict)

    def reload_vae_model(self, vae_path, vae_file_format='fp32'):
        if len(vae_path) > 0 and vae_path[-1] != "/":
            vae_path = vae_path + "/"
        return self.model.reload_vae_model(vae_path, vae_file_format)

    def convert_state_dict(self, state_dict):
        # replace deprecated weights
        for path in ["encoder.mid_block.attentions.0", "decoder.mid_block.attentions.0"]:
            # group_norm path stays the same

            # query -> to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(
                    f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(
                    f"{path}.query.bias")

            # key -> to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(
                    f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(
                    f"{path}.key.bias")

            # value -> to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(
                    f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(
                    f"{path}.value.bias")

            # proj_attn -> to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(
                    f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(
                    f"{path}.proj_attn.bias")

        for key in state_dict:
            # print(key)
            if len(state_dict[key].shape) == 4:
                state_dict[key] = state_dict[key].permute(
                    0, 2, 3, 1).contiguous()
            else:
                state_dict[key] = state_dict[key]
            if self.is_upcast and (key.startswith("decoder.up_blocks.2") or key.startswith("decoder.up_blocks.3") or key.startswith("decoder.conv_norm_out")):
                # print(key)
                state_dict[key] = state_dict[key].to(torch.float32)
            else:
                state_dict[key] = state_dict[key].to(torch.float16)

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.reload_vae_model_from_cache(state_dict, "cpu")

    def load_from_diffusers_model(self, model_path, cache=False):
        checkpoint_file = os.path.join(
            model_path, "diffusion_pytorch_model.bin")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(
                model_path, "diffusion_pytorch_model.safetensors")
        if checkpoint_file in self.vae_cache:
            state_dict = self.vae_cache[checkpoint_file]
        else:
            if "safetensors" in checkpoint_file:
                state_dict = load_file(checkpoint_file)
            else:
                state_dict = torch.load(checkpoint_file, map_location="cpu")

            state_dict = self.convert_state_dict(state_dict)

            if cache:
                self.vae_cache[checkpoint_file] = state_dict

        return self.load_state_dict(state_dict)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def lyra_decode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.model.vae_decode(x, self.map_extra_tensors, self.scale_params)
        # print(x)
        return x.permute(0, 3, 1, 2)

    def lyra_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.model.vae_encode(x, self.map_extra_tensors, self.scale_params)
        return x.permute(0, 3, 1, 2)

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True, map_extra_tensors=None, scale_params=None,
    ) -> DiagonalGaussianDistribution:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        self.set_s3param(False)
        x = x.to(torch.float16)
        self.map_extra_tensors = map_extra_tensors
        self.scale_params = scale_params

        if not self.is_dynamic_tiling_s3diff:
            if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                return self.tiled_encode(x, return_dict=return_dict)
        else:
            H,W = x.shape[2], x.shape[3]
            if self.use_tiling and max(H, W) > self.pad * 2 + self.tile_size:
                print("[Tiled VAE encode]: the input size is not tiny and necessary to tile.", H, W, self.pad, self.tile_size, self.pad * 2 + self.tile_size)
                return self.tiled_encode(x, return_dict=return_dict)
            else:
                print("[Tiled VAE encode]: the input size is tiny and unnecessary to tile.", H, W, self.pad, self.tile_size, self.pad * 2 + self.tile_size)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.lyra_encode(
                x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
            posterior = DiagonalGaussianDistribution(h)
        else:
            moments = self.lyra_encode(x)
            posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        if not self.is_dynamic_tiling_s3diff:
            if self.use_tiling and (z.shape[2] > self.tile_latent_min_size or z.shape[3] > self.tile_latent_min_size):
                return self.tiled_decode(z, return_dict=return_dict)
        else:
            H,W = z.shape[2],z.shape[3]
            if self.use_tiling and max(H, W) > self.pad * 2 + self.tile_size:
                print("[Tiled VAE decode]: the input size is not tiny and necessary to tile.", H, W, self.pad, self.tile_size, self.pad * 2 + self.tile_size)
                return self.tiled_decode(z, return_dict=return_dict)

        print("[Tiled VAE decode]: the input size is tiny and unnecessary to tile.", H, W, self.pad, self.tile_size, self.pad * 2 + self.tile_size)
        dec = self.lyra_decode(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None, map_extra_tensors=None, scale_params=None,
    ) -> torch.FloatTensor:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        self.set_s3param(True)
        z = z.to(torch.float16)
        self.map_extra_tensors = map_extra_tensors
        self.scale_params = scale_params
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(
                z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * \
                (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * \
                (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    # ====================== for s3diff
    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def set_tile_pad(self, tile, pad, is_decode):
        if is_decode:
            self.decode_tile_size = tile
            self.decode_pad = pad
        else:
            self.encode_tile_size = tile
            self.encode_pad = pad

    def set_s3param(self, is_decoder):
        if is_decoder:
            self.tile_size = self.decode_tile_size
            self.pad = self.decode_pad
        else:
            self.tile_size = self.encode_tile_size
            self.pad = self.encode_pad

    def split_tiles(self, h, w, is_decoder):
        """
        Tool function to split the image into tiles
        @param h: height of the image
        @param w: width of the image
        @return: tile_input_bboxes, tile_output_bboxes
        """
        if not is_decoder:
            self.tile_size = 1024
            self.pad = 32
        else:
            self.tile_size = 224
            self.pad = 11
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x1, x2, y1, y2]
                # the padding is is unnessary for image borders. So we directly start from (32, 32)
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                # if the output bbox is close to the image boundary, we extend it to the image boundary
                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the final output bbox
                output_bbox = [x * 8 if is_decoder else x // 8 for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                # indistinguishable expand the input bbox by pad pixels
                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    min(w, input_bbox[1] + pad),
                    max(0, input_bbox[2] - pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes


    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        print("tiled_decode", z.shape)
        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        in_bboxes, out_bboxes = self.split_tiles(height, width, True)

        result = None
        for i in range(len(in_bboxes)):
            input_bbox = in_bboxes[i]
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]] #.cpu()
            if result is None:
                result = torch.zeros([N, 3, out_bboxes[-1][3], out_bboxes[-1][1]]).cuda().to(torch.float16)
            decoded = self.lyra_decode(tile)
            decoded_crop = crop_valid_region(decoded, in_bboxes[i], out_bboxes[i], True)
            result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = decoded_crop

        return DecoderOutput(sample=result)


    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> DiagonalGaussianDistribution:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """

        z = x
        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        in_bboxes, out_bboxes = self.split_tiles(height, width, False)
        result = None
        tiles = []
        for i in range(len(in_bboxes)):
            input_bbox = in_bboxes[i]
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]]
            if result is None:
                result = torch.zeros([N, 8, out_bboxes[-1][3], out_bboxes[-1][1]]).cuda().to(torch.float16)
            decoded = self.lyra_encode(tile)
            crop_decoded = crop_valid_region(decoded, in_bboxes[i], out_bboxes[i], False)

            tiles.append(tile)
            result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = crop_decoded

        posterior = DiagonalGaussianDistribution(result)
        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)