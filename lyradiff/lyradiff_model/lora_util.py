import os
import re
import time
import torch
import numpy as np
from safetensors.torch import load_file
from diffusers.loaders import LoraLoaderMixin
from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers, _convert_non_diffusers_lora_to_diffusers
from types import SimpleNamespace
from .lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
import logging.handlers
LORA_PREFIX_UNET = "lora_unet"
from .lyradiff_utils import LyraQuantLevel
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_UNET_LAYERS = ['lora_unet_down_blocks_0_attentions_0', 'lora_unet_down_blocks_0_attentions_1', 'lora_unet_down_blocks_1_attentions_0', 'lora_unet_down_blocks_1_attentions_1', 'lora_unet_down_blocks_2_attentions_0', 'lora_unet_down_blocks_2_attentions_1', 'lora_unet_mid_block_attentions_0', 'lora_unet_up_blocks_1_attentions_0',
                    'lora_unet_up_blocks_1_attentions_1', 'lora_unet_up_blocks_1_attentions_2', 'lora_unet_up_blocks_2_attentions_0', 'lora_unet_up_blocks_2_attentions_1', 'lora_unet_up_blocks_2_attentions_2', 'lora_unet_up_blocks_3_attentions_0', 'lora_unet_up_blocks_3_attentions_1', 'lora_unet_up_blocks_3_attentions_2']


def add_text_lora_layer(clip_model, lora_model_path="Misaka.safetensors", alpha=1.0, lora_file_format="fp32", device="cuda:0"):
    if lora_file_format == "fp32":
        model_dtype = np.float32
    elif lora_file_format == "fp16":
        model_dtype = np.float16
    else:
        raise Exception(f"unsupported model dtype: {lora_file_format}")
    all_files = os.scandir(lora_model_path)
    unload_dict = []
    # directly update weight in diffusers model
    for file in all_files:

        if 'text' in file.name:
            layer_infos = file.name.split('.')[0].split(
                'text_model_')[-1].split('_')
            curr_layer = clip_model.text_model
        else:
            continue

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                    # if temp_name == "self":
                    #     temp_name += "_" + layer_infos.pop(0)
                    # elif temp_name != "mlp" and len(layer_infos) == 1:
                    #     temp_name += "_" + layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        data = torch.from_numpy(np.fromfile(file.path, dtype=model_dtype)).to(
            clip_model.dtype).to(clip_model.device).reshape(curr_layer.weight.data.shape)
        if len(curr_layer.weight.data) == 4:
            adding_weight = alpha * data.permute(0, 3, 1, 2)
        else:
            adding_weight = alpha * data
        curr_layer.weight.data += adding_weight

        curr_layer_unload_data = {
            "layer": curr_layer,
            "added_weight": adding_weight
        }
        unload_dict.append(curr_layer_unload_data)
    return unload_dict


def add_xltext_lora_layer(clip_model, clip_model_2, lora_model_path, alpha=1.0, lora_file_format="fp32", device="cuda:0"):
    if lora_file_format == "fp32":
        model_dtype = np.float32
    elif lora_file_format == "fp16":
        model_dtype = np.float16
    else:
        raise Exception(f"unsupported model dtype: {lora_file_format}")
    all_files = os.scandir(lora_model_path)
    unload_dict = []
    # directly update weight in diffusers model
    for file in all_files:

        if 'text' in file.name:
            layer_infos = file.name.split('.')[0].split(
                'text_model_')[-1].split('_')
            if "text_encoder_2" in file.name:
                curr_layer = clip_model_2.text_model
            elif "text_encoder" in file.name:
                curr_layer = clip_model.text_model
            else:
                raise ValueError(
                    "Cannot identify clip model, need text_encoder or text_encoder_2 in filename, found: ", file.name)
        else:
            continue

        # find the target layer
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                    # if temp_name == "self":
                    #     temp_name += "_" + layer_infos.pop(0)
                    # elif temp_name != "mlp" and len(layer_infos) == 1:
                    #     temp_name += "_" + layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        data = torch.from_numpy(np.fromfile(file.path, dtype=model_dtype)).to(
            clip_model.dtype).to(clip_model.device).reshape(curr_layer.weight.data.shape)
        if len(curr_layer.weight.data) == 4:
            adding_weight = alpha * data.permute(0, 3, 1, 2)
        else:
            adding_weight = alpha * data
        curr_layer.weight.data += adding_weight

        curr_layer_unload_data = {
            "layer": curr_layer,
            "added_weight": adding_weight
        }
        unload_dict.append(curr_layer_unload_data)
    return unload_dict

def lora_trans(state_dict):
    loraload = LoraLoaderMixin()
    unet_config = SimpleNamespace(**{'layers_per_block': 2})
    state_dicts = _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)
    state_dicts_trans, state_dicts_alpha = _convert_non_diffusers_lora_to_diffusers(
        state_dicts)
    keys = list(state_dicts_trans.keys())
    for k in keys:
        key = k.replace('processor.', '')
        for x in ['.lora_linear_layer.', '_lora.', '.lora.']:
            key = key.replace(x, '.lora_')
        if key.find('text_encoder') >= 0:
            for x in ['q', 'k', 'v', 'out']:
                key = key.replace(f'.to_{x}.', f'.{x}_proj.')
        key = key.replace('to_out.', 'to_out.0.')
        if key != k:
            state_dicts_trans[key] = state_dicts_trans.pop(k)
    alpha = torch.Tensor(list(set(list(state_dicts_alpha.values()))))
    state_dicts_trans.update({'lora.alpha': alpha})

    return state_dicts_trans


def load_lora_state_dict(filename, need_trans=True, is_transformer=False):
    state_dict = load_file(os.path.abspath(filename), device="cpu")
    if need_trans:
        if is_transformer:
            state_dict = lora_trans_transformer(state_dict)
        else:
            state_dict = lora_trans(state_dict)
    return state_dict


def move_state_dict_to_cuda(state_dict):
    if isinstance(state_dict, list):
        ret_state_dict_list = []
        for s in state_dict:
            ret_state_dict_list.append(move_state_dict_to_cuda(s))
        return ret_state_dict_list
    
    ret_state_dict = {}
    for item in state_dict:
        ret_state_dict[item] = state_dict[item].cuda()
    return ret_state_dict


def get_unet_lora(state_dict, key, lora_alpha):
    layer_infos = key
    layer_infos = layer_infos.replace(".lora_up.weight", "")
    layer_infos = layer_infos.replace(".lora_down.weight", "")

    layer_infos = layer_infos[5:]
    layer_names = layer_infos.split(".")

    layers = []
    i = 0
    while i < len(layer_names):

        if len(layers) >= 4:
            layers[-1] += "_" + layer_names[i]
        elif i + 1 < len(layer_names) and layer_names[i+1].isdigit():
            layers.append(layer_names[i] + "_" + layer_names[i+1])
            i += 1
        elif len(layers) > 0 and "samplers" in layers[-1]:
            layers[-1] += "_" + layer_names[i]
        else:
            layers.append(layer_names[i])
        i += 1
    layer_infos = ".".join(layers)

    pair_keys = [key.replace("lora_down", "lora_up"),
                 key.replace("lora_up", "lora_down")]
    weight_up, weight_down = state_dict[pair_keys[0]
                                        ], state_dict[pair_keys[1]]
    weight_scale = lora_alpha / \
        weight_up.shape[1] if lora_alpha != -1 else 1.0
    # update weight
    if len(state_dict[pair_keys[0]].shape) == 4:
        weight_up = weight_up.squeeze([2, 3]).to(torch.float32)
        weight_down = weight_down.squeeze([2, 3]).to(torch.float32)
        if len(weight_down.shape) == 4:
            curr_layer_weight = weight_scale * \
                torch.einsum('a b, b c h w -> a c h w',
                             weight_up, weight_down)
        else:
            curr_layer_weight = weight_scale * \
                torch.mm(weight_up, weight_down).unsqueeze(
                    2).unsqueeze(3)

        curr_layer_weight = curr_layer_weight.permute(0, 2, 3, 1)

    else:
        weight_up = state_dict[pair_keys[0]].to(torch.float32)
        weight_down = state_dict[pair_keys[1]].to(torch.float32)

        curr_layer_weight = weight_scale * \
            torch.mm(weight_up, weight_down)

    curr_layer_weight = curr_layer_weight.to(torch.float16).contiguous()

    return layers, curr_layer_weight, pair_keys


def add_lora_to_opt_model(state_dict, unet, clip_model, clip_model_2, alpha=1.0):
    # directly update weight in diffusers model
    state_dict = move_state_dict_to_cuda(state_dict)

    alpha_ks = list(filter(lambda x: x.find('.alpha') >= 0, state_dict))
    lora_alpha = state_dict[alpha_ks[0]].item() if len(alpha_ks) > 0 else -1

    visited = set()
    for key in state_dict:
        # print(key)
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if "text" in key:
            curr_layer = clip_model_2 if key.find(
                'text_encoder_2') >= 0 else clip_model
            # print(key)
            # print(layer_infos)
            # find the target layer
            # if is_sdxl:
            layer_infos = key.split('.')[1:]
            
            for x in layer_infos:
                try:
                    curr_layer = curr_layer.__getattr__(x)
                except Exception:
                    break
                
            # update weight
            pair_keys = [key.replace("lora_down", "lora_up"),
                         key.replace("lora_up", "lora_down")]
            weight_up, weight_down = state_dict[pair_keys[0]
                                                ], state_dict[pair_keys[1]]

            weight_scale = lora_alpha/weight_up.shape[1] if lora_alpha != -1 else 1.0

            adding_weight = torch.mm(weight_up, weight_down)
            adding_weight = alpha * weight_scale * adding_weight

            curr_layer.weight.data += adding_weight.to(torch.float16)
            # update visited list
            for item in pair_keys:
                visited.add(item)

        elif "unet" in key:
            layer_infos = key
            layer_infos = layer_infos.replace(".lora_up.weight", "")
            layer_infos = layer_infos.replace(".lora_down.weight", "")

            layer_infos = layer_infos[5:]
            layer_names = layer_infos.split(".")

            layers = []
            i = 0
            while i < len(layer_names):

                if len(layers) >= 4:
                    layers[-1] += "_" + layer_names[i]
                elif i + 1 < len(layer_names) and layer_names[i+1].isdigit():
                    layers.append(layer_names[i] + "_" + layer_names[i+1])
                    i += 1
                elif len(layers) > 0 and "samplers" in layers[-1]:
                    layers[-1] += "_" + layer_names[i]
                else:
                    layers.append(layer_names[i])
                i += 1
            layer_infos = ".".join(layers)

            pair_keys = [key.replace("lora_down", "lora_up"),
                         key.replace("lora_up", "lora_down")]
            weight_up, weight_down = state_dict[pair_keys[0]
                                            ], state_dict[pair_keys[1]]
            weight_scale = lora_alpha/weight_up.shape[1] if lora_alpha != -1 else 1.0
            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = weight_up.squeeze([2, 3]).to(torch.float32)
                weight_down = weight_down.squeeze([2, 3]).to(torch.float32)
                if len(weight_down.shape) == 4:
                    curr_layer_weight = weight_scale * \
                        torch.einsum('a b, b c h w -> a c h w',
                                     weight_up, weight_down)
                else:
                    curr_layer_weight = weight_scale * \
                        torch.mm(weight_up, weight_down).unsqueeze(
                            2).unsqueeze(3)

                curr_layer_weight = curr_layer_weight.permute(0, 2, 3, 1)

            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)

                curr_layer_weight = weight_scale * \
                    torch.mm(weight_up, weight_down)
            #

            curr_layer_weight = curr_layer_weight.to(torch.float16).contiguous()

            unet.load_lora_by_name(layers, curr_layer_weight, alpha)

            for item in pair_keys:
                visited.add(item)

def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    return diffusers_name

# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are taken from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
# All credits go to `kohya-ss`.
def _convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_B.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        alpha = sds_sd.pop(sds_key + ".alpha")
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0
                    )
                i += dims[j]

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update({k: down_weight for k in ait_down_keys})

            # up_weight is split to each split
            ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416
        else:
            # down_weight is chunked to each split
            ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})  # noqa: C416

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        remaining_keys = list(sds_sd.keys())
        te_state_dict = {}
        if remaining_keys:
            if not all(k.startswith("lora_te1") for k in remaining_keys):
                raise ValueError(f"Incompatible keys detected: \n\n {', '.join(remaining_keys)}")
            for key in remaining_keys:
                if not key.endswith("lora_down.weight"):
                    continue

                lora_name = key.split(".")[0]
                lora_name_up = f"{lora_name}.lora_up.weight"
                lora_name_alpha = f"{lora_name}.alpha"
                diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    down_weight = sds_sd.pop(key)
                    sd_lora_rank = down_weight.shape[0]
                    te_state_dict[diffusers_name] = down_weight
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = sds_sd.pop(lora_name_up)

                if lora_name_alpha in sds_sd:
                    alpha = sds_sd.pop(lora_name_alpha).item()
                    scale = alpha / sd_lora_rank

                    scale_down = scale
                    scale_up = 1.0
                    while scale_down * 2 < scale_up:
                        scale_down *= 2
                        scale_up /= 2

                    te_state_dict[diffusers_name] *= scale_down
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] *= scale_up

        if te_state_dict:
            te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}

        new_state_dict = {**ait_sd, **te_state_dict}
        return new_state_dict

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


# Adapted from https://gist.github.com/Leommm-byte/6b331a1e9bd53271210b26543a7065d6
# Some utilities were reused from
# https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
def _convert_xlabs_flux_lora_to_diffusers(old_state_dict):
    new_state_dict = {}
    orig_keys = list(old_state_dict.keys())

    def handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        down_weight = sds_sd.pop(sds_key)
        up_weight = sds_sd.pop(sds_key.replace(".down.weight", ".up.weight"))

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]

        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416

    for old_key in orig_keys:
        # Handle double_blocks
        if old_key.startswith(("diffusion_model.double_blocks", "double_blocks")):
            block_num = re.search(r"double_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.transformer_blocks.{block_num}"

            if "processor.proj_lora1" in old_key:
                new_key += ".attn.to_out.0"
            elif "processor.proj_lora2" in old_key:
                new_key += ".attn.to_add_out"
            # Handle text latents.
            elif "processor.qkv_lora2" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.add_q_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_k_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_v_proj",
                    ],
                )
                # continue
            # Handle image latents.
            elif "processor.qkv_lora1" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.transformer_blocks.{block_num}.attn.to_v",
                    ],
                )
                # continue

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        # Handle single_blocks
        elif old_key.startswith(("diffusion_model.single_blocks", "single_blocks")):
            block_num = re.search(r"single_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.single_transformer_blocks.{block_num}"

            if "proj_lora" in old_key:
                new_key += ".proj_out"
            elif "qkv_lora" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [f"transformer.single_transformer_blocks.{block_num}.norm.linear"],
                )

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        else:
            # Handle other potential key patterns here
            new_key = old_key

        # Since we already handle qkv above.
        if "qkv" not in old_key:
            new_state_dict[new_key] = old_state_dict.pop(old_key)

    if len(old_state_dict) > 0:
        raise ValueError(f"`old_state_dict` should be at this point but has: {list(old_state_dict.keys())}.")

    return new_state_dict

marked_tail = ["net.0.proj", "net.2", "to_out.0"]
skipped_key = ["add_q_proj", "add_k_proj", "add_v_proj", "to_q", "to_k", "to_v"]

def merge_transformer_qkv(state_dict):
    new_state_dict = {}
    for key, val in state_dict.items():
        if "to_q.lora_B" in key:
            q_up_key = key
            k_up_key = "to_k.lora_B".join(key.split("to_q.lora_B"))
            v_up_key = "to_v.lora_B".join(key.split("to_q.lora_B"))

            q_down_key = "to_q.lora_A".join(key.split("to_q.lora_B"))

            lora_down = state_dict[q_down_key]
            lora_up = torch.cat((state_dict[q_up_key], state_dict[k_up_key], state_dict[v_up_key]))

            qkv_up_key = "to_qkv.lora_B".join(key.split("to_q.lora_B"))
            qkv_down_key = "to_qkv.lora_A".join(key.split("to_q.lora_B"))

            new_state_dict[qkv_up_key] = lora_up
            new_state_dict[qkv_down_key] = lora_down

        if "add_q_proj.lora_B" in key:
            q_up_key = key
            k_up_key = "add_k_proj.lora_B".join(key.split("add_q_proj.lora_B"))
            v_up_key = "add_v_proj.lora_B".join(key.split("add_q_proj.lora_B"))

            q_down_key = "add_q_proj.lora_A".join(key.split("add_q_proj.lora_B"))

            lora_down = state_dict[q_down_key]
            lora_up = torch.cat((state_dict[q_up_key], state_dict[k_up_key], state_dict[v_up_key]))

            qkv_up_key = "to_added_qkv.lora_B".join(key.split("add_q_proj.lora_B"))
            qkv_down_key = "to_added_qkv.lora_A".join(key.split("add_q_proj.lora_B"))

            new_state_dict[qkv_up_key] = lora_up
            new_state_dict[qkv_down_key] = lora_down
        
        if any(k in key for k in skipped_key):
            continue

        new_state_dict[key] = val
        
    return new_state_dict

def _convert_bfl_flux_control_lora_to_diffusers(original_state_dict):
    converted_state_dict = {}
    original_state_dict_keys = list(original_state_dict.keys())
    num_layers = 19
    num_single_layers = 38
    inner_dim = 3072
    mlp_ratio = 4.0

    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    for lora_key in ["lora_A", "lora_B"]:
        ## time_text_embed.timestep_embedder <-  time_in
        converted_state_dict[
            f"time_text_embed.timestep_embedder.linear_1.{lora_key}.weight"
        ] = original_state_dict.pop(f"time_in.in_layer.{lora_key}.weight")
        if f"time_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[
                f"time_text_embed.timestep_embedder.linear_1.{lora_key}.bias"
            ] = original_state_dict.pop(f"time_in.in_layer.{lora_key}.bias")

        converted_state_dict[
            f"time_text_embed.timestep_embedder.linear_2.{lora_key}.weight"
        ] = original_state_dict.pop(f"time_in.out_layer.{lora_key}.weight")
        if f"time_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[
                f"time_text_embed.timestep_embedder.linear_2.{lora_key}.bias"
            ] = original_state_dict.pop(f"time_in.out_layer.{lora_key}.bias")

        ## time_text_embed.text_embedder <- vector_in
        converted_state_dict[f"time_text_embed.text_embedder.linear_1.{lora_key}.weight"] = original_state_dict.pop(
            f"vector_in.in_layer.{lora_key}.weight"
        )
        if f"vector_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.text_embedder.linear_1.{lora_key}.bias"] = original_state_dict.pop(
                f"vector_in.in_layer.{lora_key}.bias"
            )

        converted_state_dict[f"time_text_embed.text_embedder.linear_2.{lora_key}.weight"] = original_state_dict.pop(
            f"vector_in.out_layer.{lora_key}.weight"
        )
        if f"vector_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"time_text_embed.text_embedder.linear_2.{lora_key}.bias"] = original_state_dict.pop(
                f"vector_in.out_layer.{lora_key}.bias"
            )

        # guidance
        has_guidance = any("guidance" in k for k in original_state_dict)
        if has_guidance:
            converted_state_dict[
                f"time_text_embed.guidance_embedder.linear_1.{lora_key}.weight"
            ] = original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.weight")
            if f"guidance_in.in_layer.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[
                    f"time_text_embed.guidance_embedder.linear_1.{lora_key}.bias"
                ] = original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.bias")

            converted_state_dict[
                f"time_text_embed.guidance_embedder.linear_2.{lora_key}.weight"
            ] = original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.weight")
            if f"guidance_in.out_layer.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[
                    f"time_text_embed.guidance_embedder.linear_2.{lora_key}.bias"
                ] = original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.bias")

        # context_embedder
        converted_state_dict[f"context_embedder.{lora_key}.weight"] = original_state_dict.pop(
            f"txt_in.{lora_key}.weight"
        )
        if f"txt_in.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"context_embedder.{lora_key}.bias"] = original_state_dict.pop(
                f"txt_in.{lora_key}.bias"
            )

        # x_embedder
        converted_state_dict[f"x_embedder.{lora_key}.weight"] = original_state_dict.pop(f"img_in.{lora_key}.weight")
        if f"img_in.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"x_embedder.{lora_key}.bias"] = original_state_dict.pop(f"img_in.{lora_key}.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."

        for lora_key in ["lora_A", "lora_B"]:
            # norms
            converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mod.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm1.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mod.lin.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}norm1_context.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm1_context.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias"
                )

            # Q, K, V
            if lora_key == "lora_A":
                sample_lora_weight = original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_lora_weight])

                context_lora_weight = original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
            else:
                sample_q, sample_k, sample_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([sample_q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([sample_k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([sample_v])

                context_q, context_k, context_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.weight"] = torch.cat([context_q])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.weight"] = torch.cat([context_k])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.weight"] = torch.cat([context_v])

            if f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([sample_q_bias])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([sample_k_bias])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([sample_v_bias])

            if f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias" in original_state_dict_keys:
                context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{lora_key}.bias"] = torch.cat([context_q_bias])
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{lora_key}.bias"] = torch.cat([context_k_bias])
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{lora_key}.bias"] = torch.cat([context_v_bias])

            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.0.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.2.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mlp.2.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.0.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.0.proj.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias"
                )

            converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.2.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}ff_context.net.2.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias"
                )

            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.proj.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_out.0.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_attn.proj.{lora_key}.bias"
                )
            converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.proj.{lora_key}.weight"
            )
            if f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}attn.to_add_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias"
                )

        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."

        for lora_key in ["lora_A", "lora_B"]:
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.modulation.lin.{lora_key}.weight"
            )
            if f"single_blocks.{i}.modulation.lin.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}norm.linear.{lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.modulation.lin.{lora_key}.bias"
                )

            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)

            if lora_key == "lora_A":
                lora_weight = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([lora_weight])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    lora_bias = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias")
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([lora_bias])
            else:
                q, k, v, mlp = torch.split(
                    original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight"), split_size, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.weight"] = torch.cat([q])
                converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.weight"] = torch.cat([k])
                converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.weight"] = torch.cat([v])
                converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.weight"] = torch.cat([mlp])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict_keys:
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(
                        original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias"), split_size, dim=0
                    )
                    converted_state_dict[f"{block_prefix}attn.to_q.{lora_key}.bias"] = torch.cat([q_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{lora_key}.bias"] = torch.cat([k_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{lora_key}.bias"] = torch.cat([v_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{lora_key}.bias"] = torch.cat([mlp_bias])

            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.{lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.linear2.{lora_key}.weight"
            )
            if f"single_blocks.{i}.linear2.{lora_key}.bias" in original_state_dict_keys:
                converted_state_dict[f"{block_prefix}proj_out.{lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.linear2.{lora_key}.bias"
                )

        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )

    for lora_key in ["lora_A", "lora_B"]:
        converted_state_dict[f"proj_out.{lora_key}.weight"] = original_state_dict.pop(
            f"final_layer.linear.{lora_key}.weight"
        )
        if f"final_layer.linear.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"proj_out.{lora_key}.bias"] = original_state_dict.pop(
                f"final_layer.linear.{lora_key}.bias"
            )

        converted_state_dict[f"norm_out.linear.{lora_key}.weight"] = swap_scale_shift(
            original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.weight")
        )
        if f"final_layer.adaLN_modulation.1.{lora_key}.bias" in original_state_dict_keys:
            converted_state_dict[f"norm_out.linear.{lora_key}.bias"] = swap_scale_shift(
                original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.bias")
            )

    if len(original_state_dict) > 0:
        raise ValueError(f"`original_state_dict` should be empty at this point but has {original_state_dict.keys()=}.")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict

def lora_trans_transformer(state_dict):
    is_kohya = any(".lora_down.weight" in k for k in state_dict)
    is_bfl_control = any("query_norm.scale" in k for k in state_dict)
    is_xlabs = any("processor" in k for k in state_dict)

    if is_bfl_control:
        state_dict = _convert_bfl_flux_control_lora_to_diffusers(state_dict)
    
    if is_kohya:
        state_dict = _convert_kohya_flux_lora_to_diffusers(state_dict)
    
    if is_xlabs:
        state_dict = _convert_xlabs_flux_lora_to_diffusers(state_dict)

    state_dict = merge_transformer_qkv(state_dict)

    return state_dict

def add_lora_to_transformer_model(state_dict, transformer, clip_model, clip_model_2, alpha=1.0):
    # directly update weight in diffusers model
    state_dict = move_state_dict_to_cuda(state_dict)

    alpha_ks = list(filter(lambda x: x.find('.alpha') >= 0, state_dict))
    lora_alpha = state_dict[alpha_ks[0]].item() if len(alpha_ks) > 0 else -1

    visited = set()
    for key in state_dict:
        # print(key)
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if "transformer" in key:
            layer_infos = key
            layer_infos = layer_infos.replace(".lora_A.weight", "")
            layer_infos = layer_infos.replace(".lora_B.weight", "")

            layer_infos = layer_infos[12:]

            layers = []
            cur_tail = ""
            for tail in marked_tail:
                if tail in layer_infos:
                    cur_tail = tail

            layer_infos = layer_infos.replace(cur_tail, "")
            layer_names = layer_infos.split(".")
            # print(layer_names)

            i = 0
            while i < len(layer_names):
                if layer_names[i] == "":
                    i += 1
                    continue
                if i + 1 < len(layer_names) and layer_names[i+1].isdigit():
                    layers.append(layer_names[i] + "_" + layer_names[i+1])
                    i += 1
                else:
                    layers.append(layer_names[i])
                i += 1

            if cur_tail != "":
                layers.append(cur_tail)
            layer_infos = ".".join(layers)

            pair_keys = [key.replace("lora_A", "lora_B"),
                         key.replace("lora_B", "lora_A")]
            weight_up, weight_down = state_dict[pair_keys[0]], state_dict[pair_keys[1]]
            weight_scale = lora_alpha/weight_up.shape[1] if lora_alpha != -1 else 1.0
            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = weight_up.squeeze([2, 3]).to(torch.float32)
                weight_down = weight_down.squeeze([2, 3]).to(torch.float32)
                if len(weight_down.shape) == 4:
                    curr_layer_weight = weight_scale * \
                        torch.einsum('a b, b c h w -> a c h w',
                                     weight_up, weight_down)
                else:
                    curr_layer_weight = weight_scale * \
                        torch.mm(weight_up, weight_down).unsqueeze(
                            2).unsqueeze(3)

                curr_layer_weight = curr_layer_weight.permute(0, 2, 3, 1)

            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)

                curr_layer_weight = weight_scale * \
                    torch.mm(weight_up, weight_down)
            #
            curr_layer_weight = curr_layer_weight * alpha
            curr_layer_weight = curr_layer_weight.to(torch.bfloat16).contiguous()

            transformer.load_lora_by_name(layers, curr_layer_weight, 1.0)

            for item in pair_keys:
                visited.add(item)

        elif "text" in key:
            curr_layer = clip_model_2 if key.find(
                'text_encoder_2') >= 0 else clip_model
            # print(key)
            # print(layer_infos)
            # find the target layer
            # if is_sdxl:
            layer_infos = key.split('.')[1:]
            print(layer_infos)
            for x in layer_infos:
                try:
                    curr_layer = curr_layer.__getattr__(x)
                except Exception:
                    break
                
            # update weight
            pair_keys = [key.replace("lora_A", "lora_B"),
                         key.replace("lora_B", "lora_A")]
            weight_up, weight_down = state_dict[pair_keys[0]
                                                ], state_dict[pair_keys[1]]

            weight_scale = lora_alpha/weight_up.shape[1] if lora_alpha != -1 else 1.0

            adding_weight = torch.mm(weight_up, weight_down)
            adding_weight = alpha * weight_scale * adding_weight

            curr_layer.weight.data += adding_weight.to(torch.bfloat16)
            # update visited list
            for item in pair_keys:
                visited.add(item)


def get_curr_layer_weight(state_dict, alpha, pair_keys, dtype=torch.bfloat16):
    if pair_keys[0] not in state_dict:
        print(f"{pair_keys[0]} not in cur state dict")
        return None
    weight_up, weight_down = state_dict[pair_keys[0]], state_dict[pair_keys[1]]
    weight_scale = 1.0 # flux 暂时没有 weight scale
    # update weight
    if len(state_dict[pair_keys[0]].shape) == 4:
        weight_up = weight_up.squeeze([2, 3]).to(torch.float32)
        weight_down = weight_down.squeeze([2, 3]).to(torch.float32)
        if len(weight_down.shape) == 4:
            curr_layer_weight = weight_scale * \
                torch.einsum('a b, b c h w -> a c h w',
                                weight_up, weight_down)
        else:
            curr_layer_weight = weight_scale * \
                torch.mm(weight_up, weight_down).unsqueeze(
                    2).unsqueeze(3)

        curr_layer_weight = curr_layer_weight.permute(0, 2, 3, 1)

    else:
        weight_up = state_dict[pair_keys[0]].to(torch.float32)
        weight_down = state_dict[pair_keys[1]].to(torch.float32)

        curr_layer_weight = weight_scale * \
            torch.mm(weight_up, weight_down)
    #
    curr_layer_weight = curr_layer_weight * alpha
    curr_layer_weight = curr_layer_weight.to(dtype).contiguous()
    return curr_layer_weight

def get_merged_lora(lora_state_dicts, alphas, pair_keys, dtype=torch.bfloat16):
    curr_layer_weight = None
    # print(len(lora_state_dicts))

    for i, state_dict in enumerate(lora_state_dicts):
        # print(f"cur index: {i}")
        cur_res = get_curr_layer_weight(state_dict, alphas[i], pair_keys, dtype)
        if cur_res is None:
            continue
        
        if curr_layer_weight is None:
            curr_layer_weight = cur_res
        else:
            curr_layer_weight = curr_layer_weight + cur_res

    return curr_layer_weight.contiguous()


def add_lora_to_transformer_model_v2(lora_state_dicts, alphas, transformer, clip_model, clip_model_2, dtype=torch.bfloat16):
    # directly update weight in diffusers model
    lora_state_dicts = move_state_dict_to_cuda(lora_state_dicts)

    # alpha_ks = list(filter(lambda x: x.find('.alpha') >= 0, state_dict))
    lora_alpha = -1
    all_lora_keys = set()
    for state_dict in lora_state_dicts:
        all_lora_keys.update(state_dict.keys())

    # print(all_lora_keys)
    visited = set()
    for key in all_lora_keys:
        # print(key)
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if "transformer" in key:
            layer_infos = key
            layer_infos = layer_infos.replace(".lora_A.weight", "")
            layer_infos = layer_infos.replace(".lora_B.weight", "")

            layer_infos = layer_infos[12:]

            layers = []
            cur_tail = ""
            for tail in marked_tail:
                if tail in layer_infos:
                    cur_tail = tail

            layer_infos = layer_infos.replace(cur_tail, "")
            layer_names = layer_infos.split(".")
            # print(layer_names)

            i = 0
            while i < len(layer_names):
                if layer_names[i] == "":
                    i += 1
                    continue
                if i + 1 < len(layer_names) and layer_names[i+1].isdigit():
                    layers.append(layer_names[i] + "_" + layer_names[i+1])
                    i += 1
                else:
                    layers.append(layer_names[i])
                i += 1

            if cur_tail != "":
                layers.append(cur_tail)
            layer_infos = ".".join(layers)

            pair_keys = [key.replace("lora_A", "lora_B"),
                         key.replace("lora_B", "lora_A")]
            
            curr_layer_weight = get_merged_lora(lora_state_dicts, alphas, pair_keys, dtype)
            if isinstance(transformer, LyraDiffFluxTransformer2DModelV2):
                l_idx = int(layers[0].split("_")[-1])
                if "single" in layers[0]:
                    transformer.single_transformer_blocks[l_idx].load_lora(layers[1:], curr_layer_weight, 1.0)
                else:
                    transformer.transformer_blocks[l_idx].load_lora(layers[1:], curr_layer_weight, 1.0)
            else:
                transformer.load_lora(layers, curr_layer_weight, 1.0)

            for item in pair_keys:
                visited.add(item)

        elif "text" in key:
            curr_layer = clip_model_2 if key.find(
                'text_encoder_2') >= 0 else clip_model
            # print(key)
            # print(layer_infos)
            # find the target layer
            # if is_sdxl:
            layer_infos = key.split('.')[1:]
            print(layer_infos)
            for x in layer_infos:
                try:
                    curr_layer = curr_layer.__getattr__(x)
                except Exception:
                    break
                
            # update weight
            pair_keys = [key.replace("lora_A", "lora_B"),
                         key.replace("lora_B", "lora_A")]
            
            curr_layer_weight = get_merged_lora(lora_state_dicts, alphas, pair_keys, dtype)

            curr_layer.weight.data += curr_layer_weight.to(dtype)
            # update visited list
            for item in pair_keys:
                visited.add(item)

# load lora for flux
def flux_load_lora(lora_file_list, alpha_list, transformer_model, text_encoder, text_encoder_2, quant_level=LyraQuantLevel.NONE):
    start = time.perf_counter()
    lora_state_dict_list = []
    for lora_file in lora_file_list:
        lora_state_dict_list.append(load_lora_state_dict(lora_file, need_trans=True, is_transformer=True))

    # 将加载好的lora state dict 加载入到unet以及clip模型中
    add_lora_to_transformer_model_v2(lora_state_dict_list, alpha_list, transformer_model, text_encoder, text_encoder_2)
    print(f"lora load time cost: {time.perf_counter() - start}")
    return lora_state_dict_list

def flux_clear_lora(lora_state_dict_list, alpha_list, transformer_model, text_encoder, text_encoder_2, quant_level=LyraQuantLevel.NONE):
    neg_alpha_list = [a * -1.0 for a in alpha_list]
    if quant_level != LyraQuantLevel.NONE:
        # fp8下清除lora的方法
        if isinstance(transformer_model, lyradiffFluxTransformer2DModelV2):
            for model in transformer_model.single_transformer_blocks:
                model.clear_lora()
            for model in transformer_model.transformer_blocks:
                model.clear_lora()
        else:
            transformer_model.clear_lora()
    else:
        add_lora_to_transformer_model_v2(lora_state_dict_list, neg_alpha_list, transformer_model, text_encoder, text_encoder_2)
