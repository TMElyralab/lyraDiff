import argparse
import os

import torch
import time
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel


def conver_diffusers_flux_to_opt_model(args):
    checkpoint_path = os.path.abspath(args.diffusers_model_path)

    print(f"Load diffusers flux model from {checkpoint_path}")

    model = FluxTransformer2DModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    # print(model.norm_out.norm)
    checkpoint = model.state_dict()

    output_dir = os.path.abspath(os.path.join(
        args.optimized_model_dir, 'transformer_bins'))
    if args.fp16:
        output_dir = os.path.abspath(os.path.join(
            args.optimized_model_dir, 'transformer_bins_fp16'))
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
        description="Conver diffusers unet model bin file to our optimized model parameter files")
    parser.add_argument("--diffusers-model-path", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    start = time.perf_counter()
    conver_diffusers_flux_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")

# python3 convert_unet_bin.py --diffusers-ckpt-file=/cfs-datasets/public_models/stable-diffusion-inpainting/unet/diffusion_pytorch_model.fp16.bin --optmized-model-dir=/cfs-datasets/public_models/stable-diffusion-inpainting --fp16