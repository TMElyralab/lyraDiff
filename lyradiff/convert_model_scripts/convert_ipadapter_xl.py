import argparse
import os,sys

import torch
import time


s = """
['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.processor', 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor', 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor', 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.processor', 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn2.processor', 'mid_block.attentions.0.transformer_blocks.1.attn1.processor', 'mid_block.attentions.0.transformer_blocks.1.attn2.processor', 'mid_block.attentions.0.transformer_blocks.2.attn1.processor', 'mid_block.attentions.0.transformer_blocks.2.attn2.processor', 'mid_block.attentions.0.transformer_blocks.3.attn1.processor', 'mid_block.attentions.0.transformer_blocks.3.attn2.processor', 'mid_block.attentions.0.transformer_blocks.4.attn1.processor', 'mid_block.attentions.0.transformer_blocks.4.attn2.processor', 'mid_block.attentions.0.transformer_blocks.5.attn1.processor', 'mid_block.attentions.0.transformer_blocks.5.attn2.processor', 'mid_block.attentions.0.transformer_blocks.6.attn1.processor', 'mid_block.attentions.0.transformer_blocks.6.attn2.processor', 'mid_block.attentions.0.transformer_blocks.7.attn1.processor', 'mid_block.attentions.0.transformer_blocks.7.attn2.processor', 'mid_block.attentions.0.transformer_blocks.8.attn1.processor', 'mid_block.attentions.0.transformer_blocks.8.attn2.processor', 'mid_block.attentions.0.transformer_blocks.9.attn1.processor', 'mid_block.attentions.0.transformer_blocks.9.attn2.processor']
"""

attn_list = eval(s)
attn2_list = []
for name in attn_list:
    if "attn2" in name:
        attn2_list.append(".".join(name.split(".")[:-1]))
print(len(attn2_list))

mapping =  [line.split(" ") for line in s.strip().split("\n")]

def conver_diffusers_ipadater(args):
    dir_output = os.path.join(args.model_dir, "lyra_tran", args.subfolder)
    os.makedirs(dir_output, exist_ok=True)

    fpath_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    print(f"parse ip adapter weight file: {fpath_weight}")
    state_dict = torch.load(fpath_weight, map_location="cpu")

    key_adapter = "ip_adapter"

    basename_without_ext = os.path.splitext(os.path.basename(args.weight_name))[0]

    dir_output_ipadapter = os.path.join(dir_output, basename_without_ext)
    os.makedirs(dir_output_ipadapter, exist_ok=True)
    key = 1
    # for item in mapping:
    for name in attn2_list:
        fpath_out = os.path.join(dir_output_ipadapter, f"{name}.to_k_ip.weight.bin")
        state_dict[key_adapter][f"{key}.to_k_ip.weight"].numpy().tofile(fpath_out)
        state_dict[key_adapter][f"{key}.to_v_ip.weight"].numpy().tofile(fpath_out.replace("_k_", "_v_"))
        key += 2

    fpath_ori_weight = os.path.join(args.model_dir, args.subfolder, args.weight_name)
    fpath_image_encoder = os.path.join(args.model_dir, args.subfolder, "image_encoder")
    os.system("ln -s {} {}".format(fpath_ori_weight, dir_output))
    os.system("ln -s {} {}".format(fpath_image_encoder, dir_output))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ipadater bin file to our optimized model parameter files")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--subfolder", type=str, required=True)
    parser.add_argument("--weight_name", type=str, required=True)
    args = parser.parse_args()
    conver_diffusers_ipadater(args)


# python3 convert_facein.py --model_dir=/cfs-datasets/projects/VirtualIdol/models/ip_adapter/ --subfolder=sdxl_models --weight_name=ip-adapter-plus_sdxl_vit-h.bin 
