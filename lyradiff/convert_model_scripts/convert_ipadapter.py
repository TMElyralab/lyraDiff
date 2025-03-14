import argparse
import os

import torch
import time
from safetensors.torch import load_file


s = """
down_blocks.0.attentions.0.transformer_blocks.0.attn2 1
down_blocks.0.attentions.1.transformer_blocks.0.attn2 3
down_blocks.1.attentions.0.transformer_blocks.0.attn2 5
down_blocks.1.attentions.1.transformer_blocks.0.attn2 7
down_blocks.2.attentions.0.transformer_blocks.0.attn2 9
down_blocks.2.attentions.1.transformer_blocks.0.attn2 11
up_blocks.1.attentions.0.transformer_blocks.0.attn2 13
up_blocks.1.attentions.1.transformer_blocks.0.attn2 15
up_blocks.1.attentions.2.transformer_blocks.0.attn2 17
up_blocks.2.attentions.0.transformer_blocks.0.attn2 19
up_blocks.2.attentions.1.transformer_blocks.0.attn2 21
up_blocks.2.attentions.2.transformer_blocks.0.attn2 23
up_blocks.3.attentions.0.transformer_blocks.0.attn2 25
up_blocks.3.attentions.1.transformer_blocks.0.attn2 27
up_blocks.3.attentions.2.transformer_blocks.0.attn2 29
mid_block.attentions.0.transformer_blocks.0.attn2 31
"""

mapping =  [line.split(" ") for line in s.strip().split("\n")]

def load_ip_adater_file(fpath_weight, is_safetensors):
    if is_safetensors:
        state_dict = load_file(fpath_weight)
    else:
        state_dict = torch.load(fpath_weight, map_location="cpu")
    return state_dict

def conver_diffusers_ipadater_base(args):
    dir_output = os.path.join(args.model_dir, "lyra_tran", args.subfolder)
    os.makedirs(dir_output, exist_ok=True)

    fpath_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    print(f"parse ip adapter weight file: {fpath_weight}")
    state_dict = load_ip_adater_file(fpath_weight, args.safetensors)

    print(state_dict["image_proj"].keys())
    print(state_dict["ip_adapter"].keys())
    basename_without_ext = os.path.splitext(os.path.basename(args.weight_name))[0]

    dir_output_ipadapter = os.path.join(dir_output, basename_without_ext)
    os.makedirs(dir_output_ipadapter, exist_ok=True)
    for item in mapping:
        name, key = item[0], item[1]
        fpath_out = os.path.join(dir_output_ipadapter, f"{name}.to_k_ip.weight.bin")
        state_dict["ip_adapter"][f"{key}.to_k_ip.weight"].numpy().tofile(fpath_out)
        state_dict["ip_adapter"][f"{key}.to_v_ip.weight"].numpy().tofile(fpath_out.replace("to_k_ip", "to_v_ip"))

    fpath_ori_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    fpath_image_encoder = os.path.join(args.model_dir, args.subfolder, "image_encoder")
    os.system("ln -s {} {}".format(fpath_ori_weight, dir_output))
    os.system("ln -s {} {}".format(fpath_image_encoder, dir_output))
    
    for key in state_dict["image_proj"]:
        fpath_weight_out = os.path.join(dir_output, "{}.{}.bin".format(basename_without_ext, key))
        print(state_dict["image_proj"][key].shape)
        print(fpath_weight_out)
        val = state_dict['image_proj'][key].numpy()
        print(val.shape, val.dtype)
        val.tofile(fpath_weight_out)

# SDXL
s_xl = """
['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn2.processor', 'mid_block.attentions.0.transformer_blocks.1.attn1.processor', 'mid_block.attentions.0.transformer_blocks.1.attn2.processor', 'mid_block.attentions.0.transformer_blocks.2.attn1.processor', 'mid_block.attentions.0.transformer_blocks.2.attn2.processor', 'mid_block.attentions.0.transformer_blocks.3.attn1.processor', 'mid_block.attentions.0.transformer_blocks.3.attn2.processor', 'mid_block.attentions.0.transformer_blocks.4.attn1.processor', 'mid_block.attentions.0.transformer_blocks.4.attn2.processor', 'mid_block.attentions.0.transformer_blocks.5.attn1.processor', 'mid_block.attentions.0.transformer_blocks.5.attn2.processor', 'mid_block.attentions.0.transformer_blocks.6.attn1.processor', 'mid_block.attentions.0.transformer_blocks.6.attn2.processor', 'mid_block.attentions.0.transformer_blocks.7.attn1.processor', 'mid_block.attentions.0.transformer_blocks.7.attn2.processor', 'mid_block.attentions.0.transformer_blocks.8.attn1.processor', 'mid_block.attentions.0.transformer_blocks.8.attn2.processor', 'mid_block.attentions.0.transformer_blocks.9.attn1.processor', 'mid_block.attentions.0.transformer_blocks.9.attn2.processor']
"""
attn_list = eval(s_xl)
attn2_list = []
for name in attn_list:
    if "attn2" in name:
        attn2_list.append(".".join(name.split(".")[:-1]))
print(len(attn2_list))

def convert_diffusers_ipadapter_xl(args):
    dir_output = os.path.join(args.model_dir, "lyra_tran", args.subfolder)
    os.makedirs(dir_output, exist_ok=True)

    fpath_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    print(f"parse ip adapter weight file: {fpath_weight}")

    state_dict = load_ip_adater_file(fpath_weight, args.safetensors)

    key_adapter = "ip_adapter"

    basename_without_ext = os.path.splitext(os.path.basename(args.weight_name))[0]

    dir_output_ipadapter = os.path.join(dir_output, basename_without_ext)
    os.makedirs(dir_output_ipadapter, exist_ok=True)
    key = 1
    
    for name in attn2_list:
        fpath_out = os.path.join(dir_output_ipadapter, f"{name}.to_k_ip.weight.bin")
        state_dict[key_adapter][f"{key}.to_k_ip.weight"].numpy().tofile(fpath_out)
        state_dict[key_adapter][f"{key}.to_v_ip.weight"].numpy().tofile(fpath_out.replace("_k_", "_v_"))
        key += 2

    fpath_ori_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    fpath_image_encoder = os.path.join(args.model_dir, args.subfolder, "image_encoder")
    os.system("ln -s {} {}".format(fpath_ori_weight, dir_output))
    os.system("ln -s {} {}".format(fpath_image_encoder, dir_output))



def conver_diffusers_ipadater(args):
    if args.is_xl:
        convert_diffusers_ipadapter_xl(args)
    else:
        conver_diffusers_ipadater_base(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ipadater bin file to our optimized model parameter files")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--subfolder", type=str, required=True)
    parser.add_argument("--weight_name", type=str, required=True)
    parser.add_argument("--safetensors", action="store_true")
    parser.add_argument("--is_xl", action="store_true")
    args = parser.parse_args()
    conver_diffusers_ipadater(args)

# python3 convert_ipadapter.py --model_dir=/cfs-datasets/public_models/IP-Adapter --subfolder=models --weight_name=ip-adapter_sd15.bin
