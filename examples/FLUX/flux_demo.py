import torch
import time
from diffusers import FluxPipeline
import GPUtil
import os
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
from lyradiff.lyradiff_model.lora_util import flux_load_lora, flux_clear_lora
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel

model_path = "/path/to/lyraDiff-FLUX.1-dev/"

quant_level = LyraQuantLevel.INT4_W4A4_FULL
quant_level = LyraQuantLevel.FP8_W8A8_FULL
quant_level = LyraQuantLevel.NONE

# for int4, please use LyraDiffFluxTransformer2DModel for now
# transformer_model = LyraDiffFluxTransformer2DModel(quant_level=quant_level)
transformer_model = LyraDiffFluxTransformer2DModelV2(quant_level=quant_level)

start = time.perf_counter()
transformer_model.load_from_diffusers_model(os.path.join(model_path, "transformer"))

# for int4 only
# transformer_model.load_config(os.path.join(model_path, "transformer/config.json"))
# transformer_model.load_from_bin(os.path.join(model_path, "transformer_bins_fp8xint4"))

print(f"after load transformer_model: {time.perf_counter() - start}")
GPUtil.showUtilization(all=True)

model = FluxPipeline.from_pretrained(
    model_path,
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
model.transformer = transformer_model

# lora_file_list1 = ["/path/to/lora.safetensors"]
# lora_alpha_list1 = [0.5]

# lora_state_dict_list = flux_load_lora(lora_file_list1, lora_alpha_list1, transformer_model, model.text_encoder, model.text_encoder_2, quant_level)

prompts = ["Female furry Pixie with text hello world", "Female furry Pixie with text hello world", "Female furry Pixie with text hello world"]

# Image Gen
generator=torch.Generator("cuda").manual_seed(123)

for i in range(len(prompts)):
    # generator = torch.Generator("cuda").manual_seed(123)
    start = time.perf_counter()
    images = model(prompts[i],
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=20,
                max_sequence_length=512,
                generator=generator
            )[0]
    print("cur cost: ", time.perf_counter() - start)
    if quant_level.value == 2:
        images[0].save(f"lyradiff_flux_quant_fp8_full_{i}.png")
    elif quant_level.value == 1:
        images[0].save(f"lyradiff_flux_quant_fp8_{i}.png")
    elif quant_level.value == 8:
        images[0].save(f"lyradiff_flux_quant_int4_{i}.png")
    else:
        images[0].save(f"lyradiff_flux_quant_none_{i}.png")
    
GPUtil.showUtilization(all=True)
