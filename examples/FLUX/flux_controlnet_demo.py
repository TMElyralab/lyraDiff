import torch
import time
from diffusers import FluxControlNetModel, FluxControlNetPipeline
import GPUtil
import os
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel
from diffusers.utils import load_image

model_path = "/path/to/LyraDiff-FLUX.1-dev/"

controlnet = FluxControlNetModel.from_pretrained(
  "black-forest-labs/FLUX.1-Depth-dev",
  torch_dtype=torch.bfloat16,
  use_safetensors=True,
).to(torch.device("cuda"))

quant_level = LyraQuantLevel.NONE
transformer_model = LyraDiffFluxTransformer2DModelV2(quant_level=quant_level)

transformer_model.load_from_diffusers_model(os.path.join(model_path, "transformer"))

model = FluxControlNetPipeline.from_pretrained(
    model_path,
    controlnet=controlnet,
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")

model.transformer = transformer_model

prompts = ["Female furry Pixie with text hello world", "Female furry Pixie with text hello world", "Female furry Pixie with text hello world"]

generator=torch.Generator("cuda").manual_seed(123)
control_image = load_image("https://huggingface.co/Xlabs-AI/flux-controlnet-depth-diffusers/resolve/main/depth_example.png")
prompt = "photo of fashion woman in the street"

generator=torch.Generator("cuda").manual_seed(123)

for i in range(3):
    start = time.perf_counter()
    image = model(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=20,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    print(f"cur image gen cost: {time.perf_counter() - start}")
    image.save(f"lyradiff-flux-dev-controlnet-fp8-{i}.png")
    GPUtil.showUtilization(all=True)
