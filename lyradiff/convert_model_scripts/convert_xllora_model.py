import os
import re
import time
import torch
import numpy as np
from safetensors.torch import load_file
from diffusers.loaders import LoraLoaderMixin
from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers, _convert_kohya_lora_to_diffusers
from types import SimpleNamespace
import logging
import logging.handlers
from diffusers import StableDiffusionXLPipeline
import argparse
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_UNET_LAYERS = ['lora_unet_down_blocks_0_attentions_0', 'lora_unet_down_blocks_0_attentions_1', 'lora_unet_down_blocks_1_attentions_0', 'lora_unet_down_blocks_1_attentions_1', 'lora_unet_down_blocks_2_attentions_0', 'lora_unet_down_blocks_2_attentions_1', 'lora_unet_mid_block_attentions_0', 'lora_unet_up_blocks_1_attentions_0', 'lora_unet_up_blocks_1_attentions_1', 'lora_unet_up_blocks_1_attentions_2', 'lora_unet_up_blocks_2_attentions_0', 'lora_unet_up_blocks_2_attentions_1', 'lora_unet_up_blocks_2_attentions_2', 'lora_unet_up_blocks_3_attentions_0', 'lora_unet_up_blocks_3_attentions_1', 'lora_unet_up_blocks_3_attentions_2']

def sdxllora_trans(state_dict):
    loraload = LoraLoaderMixin()
    unet_config = SimpleNamespace(**{'layers_per_block':2})
    state_dicts = _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)
    state_dicts_trans, state_dicts_alpha = _convert_kohya_lora_to_diffusers(state_dicts)
    keys = list(state_dicts_trans.keys())
    for k in keys:
        key = k.replace('processor.', '')
        for x in ['.lora_linear_layer.','_lora.','.lora.']:
            key = key.replace(x, '.lora_')
        if key.find('text_encoder')>=0:
            for x in ['q', 'k', 'v', 'out']:
                key = key.replace(f'.to_{x}.', f'.{x}_proj.')
        key = key.replace('to_out.', 'to_out.0.')
        if key != k:
            state_dicts_trans[key] = state_dicts_trans.pop(k)
    alpha = torch.Tensor(list(set(list(state_dicts_alpha.values()))))
    state_dicts_trans.update({'lora.alpha':alpha})
    
    return state_dicts_trans

def conver_lora_to_opt_model(args):
        # directly update weight in diffusers model
    state_dict = load_file(os.path.abspath(args.lora_file))
    state_dict = sdxllora_trans(state_dict)

    visited = []

    isExist = os.path.exists(args.optimized_model_dir)
    if not isExist:
        os.makedirs(args.optimized_model_dir)

    for key in state_dict:
        # print(key)
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
        #if 'text' in key:
        #    layer_infos = key.split('.')[0].split(
        #        LORA_PREFIX_TEXT_ENCODER+'_')[-1]
        #else:
        #layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1]
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        layer_infos = key
        layer_infos = layer_infos.replace(".lora_up.weight", "")
        layer_infos = layer_infos.replace(".lora_down.weight", "")
        layer_infos = layer_infos.replace(".", "_")
        if layer_infos.startswith("unet_"):
            layer_infos = layer_infos[5:]
        #print(layer_infos)
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
            alpha_key = key.replace('lora_down.weight', 'alpha')
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
            alpha_key = key.replace('lora_up.weight', 'alpha')
           

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(
                3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            if alpha_key in state_dict:
                weight_alpha = state_dict[alpha_key].item()
                weight_rank = weight_up.shape[1]
                weight_scale = weight_alpha / weight_rank
            else:
                weight_scale = 1.0

            # if weight_down.shape[2] == 1 and weight_down.shape[3] == 1:
            #     weight_down = weight_down.squeeze(3).squeeze(2)
            #     curr_layer_weight = weight_scale * \
            #         torch.mm(weight_up, weight_down).unsqueeze(
            #             2).unsqueeze(3).permute(0, 2, 3, 1)
            # elif weight_down.shape[2] == 3 and weight_down.shape[3] == 3:

            #     in_dim = weight_down.shape[1]
            #     lora_dim = weight_down.shape[0]
            #     out_dim = weight_up.shape[0]
            #     weight_down = weight_down.reshape(
            #         [lora_dim, in_dim, 9]).permute([2, 0, 1])

            #     curr_layer_weight = weight_scale * \
            #         (weight_up @ weight_down).permute(
            #             [1, 2, 0]).view([out_dim, in_dim, 3, 3]).permute(0, 2, 3, 1)
            weight_up, weight_down = state_dict[pair_keys[0]], state_dict[pair_keys[1]]
            weight_up = weight_up.squeeze([2, 3]).to(torch.float32)
            weight_down = weight_down.squeeze([2, 3]).to(torch.float32)
            if len(weight_down.shape) == 4:
                curr_layer_weight = weight_scale * torch.einsum('a b, b c h w -> a c h w', weight_up, weight_down)
            else:
                curr_layer_weight = weight_scale * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)

            curr_layer_weight = curr_layer_weight.permute(0, 2, 3, 1)

        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            if alpha_key in state_dict:
                weight_alpha = state_dict[alpha_key].item()
                weight_rank = weight_up.shape[1]
                weight_scale = weight_alpha / weight_rank
            else:
                weight_scale = 1.0
            curr_layer_weight = weight_scale * torch.mm(weight_up, weight_down)
        #
        if args.fp16:
            curr_layer_weight = curr_layer_weight.to(torch.float16)
        curr_layer_weight = curr_layer_weight.numpy()
        print("save layer name: ", layer_infos)
        curr_layer_weight.tofile(
            f"{args.optimized_model_dir}/{layer_infos}.bin")

        # update visited list
        for item in pair_keys:
            visited.append(item)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conver diffusers unet model bin file to our optimized model parameter files")
    parser.add_argument("--lora-file", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action='store_true')
    args = parser.parse_args()

    start = time.perf_counter()
    conver_lora_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")

