import argparse
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import time

import re

from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

import numpy as np


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|proj_in|proj_out|conv1|conv2|conv).*"
    )
    return pattern.match(name) is not None


def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="min-mean",
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            quant_config["quant_cfg"][w_name] = {"enable": False}
            quant_config["quant_cfg"][i_name] = {"enable": False}

            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 1 and "ff.net" in name)
                or (quant_level >= 2 and "to_qkv" in name)
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}

            # quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            # quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}

        # 目前暂时不做卷积优化，跳过
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            quant_config["quant_cfg"][w_name] = {"enable": False}
            quant_config["quant_cfg"][i_name] = {"enable": False}

            # quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            # quant_config["quant_cfg"][i_name] = {
            #     "num_bits": 8,
            #     "axis": None,
            #     "calibrator": (
            #         PercentileCalibrator,
            #         (),
            #         {
            #             "num_bits": 8,
            #             "axis": None,
            #             "percentile": percentile,
            #             "total_step": num_inference_steps,
            #             "collect_method": collect_method,
            #         },
            #     ),
            # }
    return quant_config


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
        ).images


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    with open(calib_data_path, "r", encoding="utf8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]


def tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction."""
    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if unsigned:
        if inputs.min() < 0.0:
            raise TypeError(
                "Negative values encountered in unsigned quantization.")

    # Computation must be in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor(
        (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        # Value quantized with amax=0 should all be 0
        scale[zero_amax_mask] = 0

    # scale, min_bound, max_bound = calculate_scale(amax, num_bits, unsigned, narrow_range)

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = (
            1.0  # Return 1 makes more sense for values quantized to 0 with amax=0
        )

    if input_dtype == torch.half:
        outputs = outputs.half()

    return outputs, scale


def calculate_scale(amax, num_bits=8, unsigned=False, narrow_range=True):
    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor(
        (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        # Value quantized with amax=0 should all be 0
        scale[zero_amax_mask] = 1

    return scale


def export_model_quant_weights(model, unet_output_dir):
    for name, module in model.named_modules():
        if hasattr(module, 'input_quantizer'):
            if filter_func(name):
                continue
            if name.endswith("attn1.to_q") or name.endswith("attn1.to_k") or name.endswith("attn1.to_v"):
                continue
            if name.endswith("attn2.to_k") or name.endswith("attn2.to_v"):
                continue

            print(f"cur quantized module: {name}")
            input_quantizer = module.input_quantizer

            input_quant_scale = calculate_scale(input_quantizer.amax, input_quantizer.num_bits, input_quantizer.unsigned, input_quantizer.narrow_range)

            input_quant_scale = 1 / input_quant_scale

            input_quantizer.pre_quant_scale.detach().cpu().float().numpy().tofile(f'{unet_output_dir}/{name}.pre_quant_scale.bin')

            input_quant_scale.detach().cpu().float().numpy().tofile(f'{unet_output_dir}/{name}.input_quant_scale.bin')

            # np.save(f'{unet_output_dir}/pre_quant_scale.npy',
            #         input_quantizer.pre_quant_scale.detach().cpu().float().numpy())


            weight_quantizer = module.weight_quantizer

            weight_int8, weight_quant_scale = tensor_quant(
            module.weight.data, weight_quantizer.amax, weight_quantizer.num_bits, weight_quantizer.unsigned, weight_quantizer.narrow_range)

            weight_int8 = weight_int8.to(torch.int8)
            weight_quant_scale = 1 / weight_quant_scale

            # np.save(f'weight_int8.npy', weight_int8.detach().cpu().numpy())
            # np.save(f'weight_quant_scale.npy', weight_quant_scale.detach().cpu().float().numpy())

            weight_int8.detach().cpu().numpy().tofile(f'{unet_output_dir}/{name}.weight_int8.bin')
            weight_quant_scale.detach().cpu().float().numpy().tofile(f'{unet_output_dir}/{name}.weight_quant_scale.bin')


def conver_sd_to_opt_model(args):
    checkpoint_file = os.path.abspath(args.sd_ckpt_file)

    cali_prompts = load_calib_prompts(args.batch_size, "./calib_prompts.txt")

    args.calib_size = args.calib_size // args.batch_size

    print(f"Load sd model from {checkpoint_file}")
    if args.sdxl:
        # model = StableDiffusionXLPipeline.from_single_file(
        #     checkpoint_file, use_safetensors=args.use_safetensors)

        model = StableDiffusionXLPipeline.from_pretrained(
            checkpoint_file, use_safetensors=args.use_safetensors, torch_dtype=torch.float16)
    else:
        model = StableDiffusionPipeline.from_pretrained(
            checkpoint_file, use_safetensors=args.use_safetensors)

    model = model.to(torch_dtype=torch.float16)

    # if args.fp16:
    #     model = model.to(torch_dtype=torch.float16)

    extra_step = (1 if not args.sdxl else 0)

    quant_config = get_int8_config(
        model.unet,
        args.quant_level,
        args.alpha,
        args.percentile,
        args.n_steps + extra_step,
        collect_method=args.collect_method,
    )

    def forward_loop(unet):
        model.unet = unet
        do_calibrate(
            pipe=model,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

    model.save_pretrained(args.optimized_model_dir, safe_serialization=True)

    model.fuse_qkv_projections()
    model = model.to("cuda")

    mtq.quantize(model.unet, quant_config, forward_loop)

    print("after model quantization and calibrate")

    unet_path = os.path.join(args.optimized_model_dir,
                             'unet/diffusion_pytorch_model.safetensors')
    unet_output_dir = os.path.abspath(
        os.path.join(args.optimized_model_dir, 'unet_bins'))
    if args.fp16:
        unet_output_dir = os.path.abspath(os.path.join(
            args.optimized_model_dir, 'unet_bins_fp16'))
    convert_to_opt_model(args, unet_path, unet_output_dir)
    export_model_quant_weights(model.unet, unet_output_dir)

    vae_path = os.path.join(args.optimized_model_dir,
                            'vae/diffusion_pytorch_model.safetensors')
    vae_output_dir = os.path.abspath(
        os.path.join(args.optimized_model_dir, 'vae_bins'))
    if args.fp16:
        vae_output_dir = os.path.abspath(os.path.join(
            args.optimized_model_dir, 'vae_bins_fp16'))

    convert_to_opt_model(args, vae_path, vae_output_dir)


def convert_to_opt_model(args, model_path, output_dir):

    print(f"Load model from {model_path}")
    checkpoint = load_file(model_path, "cpu")

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
    parser.add_argument(
        "--n_steps",
        type=int,
        default=30,
        help="Number of denoising steps, for SDXL-turbo, use 1-4 steps",
    )
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 3.0],
        help="Quantization level, 1: FFN, 2: FFN+QKV, 3: All BasicTransformerBlock",
    )
    parser.add_argument("--percentile", type=float,
                        default=1.0, required=False)

    args = parser.parse_args()

    start = time.perf_counter()
    conver_sd_to_opt_model(args)
    end = time.perf_counter()

    print(f"Convert completed, cost: {end-start} seconds.")
