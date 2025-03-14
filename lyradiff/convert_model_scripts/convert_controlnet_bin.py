import argparse
import os

import torch
import time

from safetensors.torch import load_file

def conver_diffusers_controlnet_to_opt_model(args):
    checkpoint_file = os.path.abspath(args.diffusers_ckpt_file)

    print(f"Load diffusers controlnet model from {checkpoint_file}")

    if args.safetensors:
        checkpoint = load_file(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

    output_dir = args.optimized_model_dir
    
    if not os.path.exists(output_dir):
        print(f"Create output dir: {output_dir}")
        os.makedirs(output_dir)

    for name, val in checkpoint.items():
        print(f"Parameter name: {name}, Shape: {val.shape}")

        # convert conv paramters. NCHW->NHWC format
        if "weight" in name and len(val.shape) == 4:
            val = val.permute(0, 2, 3, 1)
        if args.fp16:
            val = val.to(torch.float16)
        val = val.numpy()
        p_output_path = os.path.join(output_dir, f"{name}.bin")
        val.tofile(p_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert diffusers controlnet model bin file to our optimized model parameter files")
    parser.add_argument("--diffusers-ckpt-file", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--safetensors", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    conver_diffusers_controlnet_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")