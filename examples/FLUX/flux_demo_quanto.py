import torch
import time
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
import GPUtil
import os
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
from lyradiff.lyradiff_model.lora_util import flux_load_lora, flux_clear_lora
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel
from optimum.quanto import freeze, qfloat8, quantize, QuantizedTransformersModel

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.6 8.9 9.0"

model_path = "/path/to/lyraDiff-FLUX.1-dev/"

text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder").to(torch.bfloat16).to(torch.device("cuda"))

text_encoder_2 = T5EncoderModel.from_pretrained(
    model_path, subfolder="text_encoder_2").to(torch.bfloat16).to(torch.device("cuda"))

# use quanto to quantize text_encoder_2
quantize(text_encoder_2, weights=qfloat8) 
freeze(text_encoder_2)

torch.cuda.empty_cache()

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer")
tokenizer_2 = T5TokenizerFast.from_pretrained(
    model_path, subfolder="tokenizer_2")

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch.bfloat16).to(torch.device("cuda"))
 
# choose quant level for lyradiff
quant_level = LyraQuantLevel.INT4_W4A4_FULL
quant_level = LyraQuantLevel.FP8_W8A8_FULL

# for int4, please use LyraDiffFluxTransformer2DModel for now
# transformer_model = LyraDiffFluxTransformer2DModel(quant_level=quant_level)
transformer_model = LyraDiffFluxTransformer2DModelV2(quant_level=quant_level)

start = time.perf_counter()
transformer_model.load_from_diffusers_model(os.path.join(model_path, "transformer"))

# for int4 only
# transformer_model.load_config(os.path.join(model_path, "transformer/config.json"))
# transformer_model.load_from_bin(os.path.join(model_path, "transformer_bins_fp8xint4"))

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

model = FluxPipeline(
    vae=vae,
    transformer=transformer_model,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler
)

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
