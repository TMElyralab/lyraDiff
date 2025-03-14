import sys
sys.path.append("../..")

import torch
import time
import sys, os
from diffusers import StableDiffusionXLPipeline
from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
import GPUtil

model_path = "/path/to/sdxl/model/"
vae_model_path = "/path/to/sdxl/sdxl-vae-fp16-fix"

text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder").to(torch.float16).to(torch.device("cuda"))

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_path, subfolder="text_encoder_2").to(torch.float16).to(torch.device("cuda"))

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer_2")

unet = LyraDiffUNet2DConditionModel(is_sdxl=True)
vae = LyraDiffVaeModel(scaling_factor=0.13025, is_upcast=False)

unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(vae_model_path)

scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_path, subfolder="scheduler", timestep_spacing="linspace")

pipe = StableDiffusionXLPipeline(
    vae=vae,
    unet=unet,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler
)

ip_ckpt = "/path/to/sdxl/ip_ckpt/ip-adapter-plus_sdxl_vit-h.bin"
image_encoder_path = "/path/to/sdxl/ip_ckpt/image_encoder"

# Create LyraIPAdapter
ip_adapter = LyraIPAdapter(unet_model=unet.model, sdxl=True, device=torch.device(
    "cuda"), ip_ckpt=ip_ckpt, ip_plus=True, image_encoder_path=image_encoder_path, num_ip_tokens=16, ip_projection_dim=1024)

# load ip_adapter image
ip_image = Image.open(
    "sdxl_0.png")
ip_scale = 0.5

# get ip image embedding and pass it to the pipeline
ip_image_embedding = [ip_adapter.get_image_embeds_lyradiff(ip_image)['ip_hidden_states']]
# unet set ip adapter scale in unet model obj, since we cannot set ip_adapter_scale through diffusers pipeline
unet.set_ip_adapter_scale(ip_scale)

prompt = "a beautiful girl, cartoon style"
negative_prompt = "NSFW"
height, width = 1024, 1024
steps = 20
guidance_scale = 7.5
generator = torch.Generator("cuda").manual_seed(123)
num_images = 1

prompts = ["a beautiful girl, cartoon style"]

for i in range(len(prompts)):
    generator = torch.Generator("cuda").manual_seed(123)
    start = time.perf_counter()
    images = pipe(prompt=prompts[i],
                   height=height,
                   width=width,
                   num_inference_steps=steps,
                   num_images_per_prompt=1,
                   guidance_scale=guidance_scale,
                   negative_prompt=negative_prompt,
                   generator=generator,
                   ip_adapter_image_embeds=ip_image_embedding
                   )[0]
    print("cur cost: ", time.perf_counter() - start)
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sdxl_ip_{i}.png")
GPUtil.showUtilization(all=True)
