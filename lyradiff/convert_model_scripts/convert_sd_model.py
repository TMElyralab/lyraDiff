import argparse
import os
import diffusers
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import time

def conver_sd_to_opt_model(args):
    checkpoint_file = os.path.abspath(args.sd_ckpt_file)

    print(f"Load sd model from {checkpoint_file}")
    if args.sdxl:
        model = StableDiffusionXLPipeline.from_single_file(checkpoint_file, use_safetensors=args.use_safetensors)
    else:
        model = StableDiffusionPipeline.from_single_file(checkpoint_file, use_safetensors=args.use_safetensors)
        
    if args.fp16:
        model = model.to(torch_dtype=torch.float16)
    model.save_pretrained(args.optimized_model_dir, safe_serialization=False)
    unet_path = os.path.join(args.optimized_model_dir, 'unet/diffusion_pytorch_model.bin')
    unet_output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'unet_bins'))
    if args.fp16:
        unet_output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'unet_bins_fp16'))
    convert_to_opt_model(args, unet_path, unet_output_dir)

    vae_path = os.path.join(args.optimized_model_dir, 'vae/diffusion_pytorch_model.bin')
    vae_output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'vae_bins'))
    if args.fp16:
        vae_output_dir = os.path.abspath(os.path.join(args.optimized_model_dir, 'vae_bins_fp16'))
    
    convert_to_opt_model(args, vae_path, vae_output_dir)

    os.remove(unet_path)

def convert_to_opt_model(args, model_path, output_dir):

    print(f"Load unet model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    if not os.path.exists(output_dir):
        print(f"Create output dir: {output_dir}")
        os.makedirs(output_dir)

    for name, val in checkpoint.items():
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
    parser.add_argument("--sd-ckpt-file", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--sdxl", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    conver_sd_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")
