import torch
import time
from diffusers import StableDiffusionXLPipeline
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from lyradiff.lyradiff_model.lora_util import load_lora_state_dict, add_lora_to_opt_model
from diffusers import  EulerAncestralDiscreteScheduler

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

# lora_state = load_lora_state_dict("/path/to/sdxl/lora.safetensors")
# add_lora_to_opt_model(lora_state, unet, text_encoder, text_encoder_2, alpha=1.0)

# img gen params
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
                #    ip_adapter_image_embeds=ip_image_embedding
                   )[0]
    print("cur cost: ", time.perf_counter() - start)
    images[0].save(f"sdxl_{i}.png")
GPUtil.showUtilization(all=True)
