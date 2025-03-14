import argparse
import os

import torch
import time
from safetensors.torch import load_file
from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING, create_vae_diffusers_config, convert_ldm_vae_checkpoint, update_unet_resnet_ldm_to_diffusers, update_unet_attention_ldm_to_diffusers, create_unet_diffusers_config
from types import SimpleNamespace
import yaml
from io import BytesIO
import requests

LDM_CONTROLNET_KEY = "control_model."

def create_controlnet_diffusers_config(original_config, image_size: int):
    # unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    diffusers_unet_config = create_unet_diffusers_config(original_config, image_size=image_size)

    controlnet_config = {
        # "conditioning_channels": unet_params["hint_channels"],
        "in_channels": diffusers_unet_config["in_channels"],
        "down_block_types": diffusers_unet_config["down_block_types"],
        "block_out_channels": diffusers_unet_config["block_out_channels"],
        "layers_per_block": diffusers_unet_config["layers_per_block"],
        "cross_attention_dim": diffusers_unet_config["cross_attention_dim"],
        "attention_head_dim": diffusers_unet_config["attention_head_dim"],
        "use_linear_projection": diffusers_unet_config["use_linear_projection"],
        "class_embed_type": diffusers_unet_config["class_embed_type"],
        "addition_embed_type": diffusers_unet_config["addition_embed_type"],
        "addition_time_embed_dim": diffusers_unet_config["addition_time_embed_dim"],
        "projection_class_embeddings_input_dim": diffusers_unet_config["projection_class_embeddings_input_dim"],
        "transformer_layers_per_block": diffusers_unet_config["transformer_layers_per_block"],
    }

    return controlnet_config


def convert_controlnet_checkpoint(
    checkpoint,
    config,
):
    # Some controlnet ckpt files are distributed independently from the rest of the
    # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
    if "time_embed.0.weight" in checkpoint:
        controlnet_state_dict = checkpoint

    else:
        controlnet_state_dict = {}
        keys = list(checkpoint.keys())
        controlnet_key = LDM_CONTROLNET_KEY
        for key in keys:
            if key.startswith(controlnet_key):
                controlnet_state_dict[key.replace(controlnet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}
    ldm_controlnet_keys = DIFFUSERS_TO_LDM_MAPPING["controlnet"]["layers"]
    for diffusers_key, ldm_key in ldm_controlnet_keys.items():
        if ldm_key not in controlnet_state_dict:
            continue
        new_checkpoint[diffusers_key] = controlnet_state_dict[ldm_key]

    addition_embed_keys = DIFFUSERS_TO_LDM_MAPPING["controlnet"]["addition_embed_type"]
    for diffusers_key, ldm_key in addition_embed_keys.items():
        new_checkpoint[diffusers_key] = controlnet_state_dict[ldm_key]


    # Retrieves the keys for the input blocks only
    num_input_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "input_blocks" in layer}
    )
    input_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Down blocks
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            controlnet_state_dict,
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        if f"input_blocks.{i}.0.op.weight" in controlnet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = controlnet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = controlnet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                controlnet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {".".join(layer.split(".")[:2]) for layer in controlnet_state_dict if "middle_block" in layer}
    )
    middle_blocks = {
        layer_id: [key for key in controlnet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }
    if middle_blocks:
        resnet_0 = middle_blocks[0]
        attentions = middle_blocks[1]
        resnet_1 = middle_blocks[2]

        update_unet_resnet_ldm_to_diffusers(
            resnet_0,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.0", "new": "mid_block.resnets.0"},
        )
        update_unet_resnet_ldm_to_diffusers(
            resnet_1,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.2", "new": "mid_block.resnets.1"},
        )
        update_unet_attention_ldm_to_diffusers(
            attentions,
            new_checkpoint,
            controlnet_state_dict,
            mapping={"old": "middle_block.1", "new": "mid_block.attentions.0"},
        )

    # controlnet cond embedding blocks
    cond_embedding_blocks = {
        ".".join(layer.split(".")[:2])
        for layer in controlnet_state_dict
        if "input_hint_block" in layer and ("input_hint_block.0" not in layer) and ("input_hint_block.14" not in layer)
    }
    num_cond_embedding_blocks = len(cond_embedding_blocks)

    for idx in range(1, num_cond_embedding_blocks + 1):
        diffusers_idx = idx - 1
        cond_block_id = 2 * idx

        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.weight"] = controlnet_state_dict.pop(
            f"input_hint_block.{cond_block_id}.weight"
        )
        new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_idx}.bias"] = controlnet_state_dict.pop(
            f"input_hint_block.{cond_block_id}.bias"
        )

    # new_checkpoint["conv_in.weight"] = controlnet_state_dict['input_hint_block.0.weight']
    # new_checkpoint["conv_in.bias"] = controlnet_state_dict['input_hint_block.0.bias']

    return new_checkpoint

def convert_project_module(
    checkpoint
):
    new_checkpoint = {}
    prefix = "model.diffusion_model.project_modules."
    layer_idx = 0

    i = 11
    for key in checkpoint:
        cur_prefix = f"{prefix}{i}."
        if key.startswith(cur_prefix):
            new_name = key.replace(cur_prefix, 'mid_block.project_modules.')
            new_checkpoint[new_name] = checkpoint[key]
    i = 10
    while i >= 0:
        for k in range(4):
            if i >= 0:
                cur_prefix = f"{prefix}{i}."
                print(cur_prefix)
                print(f'up_blocks.{layer_idx}.project_modules.{k}.')

                for key in checkpoint:
                    if key.startswith(cur_prefix):
                        new_name = key.replace(cur_prefix, f'up_blocks.{layer_idx}.project_modules.{k}.')
                        new_checkpoint[new_name] = checkpoint[key]
                i -= 1

                # print(new_checkpoint.keys())

        layer_idx += 1
    print(new_checkpoint.keys())
    return new_checkpoint


def transpose_and_save(state_dict, output_dir, save_fp_16=True):
    if not os.path.exists(output_dir):
        print(f"Create output dir: {output_dir}")
        os.makedirs(output_dir)

    for name, val in state_dict.items():
        print(f"Parameter name: {name}, Shape: {val.shape}")

        # convert conv paramters. NCHW->NHWC format
        if "weight" in name and len(val.shape) == 4:
            val = val.permute(0, 2, 3, 1)

        if save_fp_16:
            val = val.to(torch.float16)
        val = val.numpy()
        p_output_path = os.path.join(output_dir, f"{name}.bin")
        val.tofile(p_output_path)

def convert_supir(args):
    original_config_file = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    original_config = yaml.safe_load(BytesIO(requests.get(original_config_file).content))
    unet_config = create_unet_diffusers_config(original_config, image_size=1024)

    vae_config = create_vae_diffusers_config(original_config, image_size=1024, scaling_factor=0.13025)
    
    print(unet_config)
    checkpoint_file = os.path.abspath(args.ckpt_file)

    print(f"Load supir model from {checkpoint_file}")

    if args.safetensors:
        checkpoint = load_file(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

    vae_denoise_state_dict = {}
    glv_controlnet_state_dict = {}
    project_modules_state_dict = {}

    for name, val in checkpoint.items():
        # print(f"Parameter name: {name}, Shape: {val.shape}")
        if "mask_LQ" in name:
            continue

        if "control_model" in name:
            glv_controlnet_state_dict[name.replace("model.", "", 1)] = val
            # glv_controlnet_state_dict[name] = val
        elif "project_modules" in name:
            project_modules_state_dict[name] = val
        else:
            vae_denoise_state_dict[name.replace("denoise_encoder", "encoder")] = val

    # print(glv_controlnet_state_dict.keys())
    # unet_config = SimpleNamespace(**{'layers_per_block':2})
    glv_controlnet_state_dict = convert_controlnet_checkpoint(glv_controlnet_state_dict, unet_config)
    project_modules_state_dict = convert_project_module(project_modules_state_dict)
    vae_denoise_state_dict = convert_ldm_vae_checkpoint(vae_denoise_state_dict, vae_config)


    output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'glv_controlnet_bins'))
    transpose_and_save(glv_controlnet_state_dict, output_dir)
    transpose_and_save(project_modules_state_dict, output_dir)

    output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'vae_denoise_encoder_bins'))
    transpose_and_save(vae_denoise_state_dict, output_dir)

    # print(vae_checkpoint.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conver diffusers unet model bin file to our optimized model parameter files")
    parser.add_argument("--ckpt-file", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--safetensors", action="store_true")

    args = parser.parse_args()

    start = time.perf_counter()
    convert_supir(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")

# python3 convert_unet_bin.py --diffusers-ckpt-file=/cfs-datasets/public_models/stable-diffusion-inpainting/unet/diffusion_pytorch_model.fp16.bin --optmized-model-dir=/cfs-datasets/public_models/stable-diffusion-inpainting --fp16