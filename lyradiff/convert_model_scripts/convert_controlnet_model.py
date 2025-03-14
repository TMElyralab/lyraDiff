import argparse
import os

import torch
import time

from diffusers import ControlNetModel

def conver_controlnet_to_opt_model(args):
    checkpoint_file = os.path.abspath(args.controlnet_ckpt_file)
    output_dir = args.optimized_model_dir

    model = ControlNetModel.from_single_file(checkpoint_file)
    if args.fp16:
        model = model.to(torch.float16)

    model.save_pretrained(output_dir, safe_serialization=False)

    controlnet_path = os.path.join(args.optimized_model_dir, 'diffusion_pytorch_model.bin')
    checkpoint = torch.load(controlnet_path, map_location="cpu")

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

    os.remove(controlnet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert controlnet model bin file to our optimized model parameter files")
    parser.add_argument("--controlnet-ckpt-file", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    conver_controlnet_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")
