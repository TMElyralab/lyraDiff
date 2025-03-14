import torch
import os

from safetensors.torch import save_file

import argparse
import time

def to_float8(x, amax, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    print(finfo.max)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    scale = 448.0 / amax
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # x_scl_sat = x * scale
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def convert_quantized_model_to_diffusers(args):
    checkpoint_path = os.path.abspath(args.quantized_model_path)

    model = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    print(f"Load quantized flux model from {checkpoint_path}")
    res = {}
    output_path = os.path.abspath(os.path.join(
        args.optimized_model_dir, 'diffusion_pytorch_model.safetensors'))

    for name, val in model["model_state_dict"].items():
        # print(f"Parameter name: {name}, Shape: {val.shape}")

        if "quantizer" in name:
            continue

        if "weight" in name:
            base_name = name.split(".weight")[0]

            weight_quantizer_name = base_name + ".weight_quantizer._amax"
            input_quantizer_name = base_name + ".input_quantizer._amax"

        if "weight" in name and weight_quantizer_name in model["model_state_dict"]:
            input_quantizer_amax = model["model_state_dict"][input_quantizer_name].float()
            print("cur name: ", name)

            input_scale = input_quantizer_amax / 448.0

            input_scale = input_scale.float().cpu()

            res[f"{base_name}_input_scale"] = input_scale
        res[name] = val.to(torch.float16)

    print(res.keys())

    save_file(res, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conver diffusers unet model bin file to our optimized model parameter files")
    parser.add_argument("--quantized-model-path", type=str, required=True)
    parser.add_argument("--optimized-model-dir", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    start = time.perf_counter()
    convert_quantized_model_to_diffusers(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")
