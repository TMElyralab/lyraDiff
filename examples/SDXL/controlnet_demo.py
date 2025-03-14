import torch
import time
from diffusers import StableDiffusionXLControlNetPipeline
import GPUtil
import os
from transformers import  CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers.pipelines.controlnet import MultiControlNetModel
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from lyradiff.lyradiff_model.lyradiff_controlnet_model import LyraDiffControlNetModel
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

model_path = "/path/to/sdxl/model/"
# We here use https://huggingface.co/madebyollin/sdxl-vae-fp16-fix to have full fp16 inference for vae
vae_model_path = "/path/to/sdxl/sdxl-vae-fp16-fix" 
controlnet_model_path = "/path/to/sdxl/controlnet/model/"

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
controlnet = LyraDiffControlNetModel(unet)

unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(vae_model_path)
controlnet.load_from_diffusers_model(
    model_name="canny", controlnet_path=controlnet_model_path)
controlnet_list = MultiControlNetModel([controlnet])
# 初始化采样器
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_path, subfolder="scheduler", timestep_spacing="linspace")

model = StableDiffusionXLControlNetPipeline(
    vae=vae,
    unet=unet,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler,
    controlnet=controlnet_list
).to(torch.float16).to(torch.device("cuda"))

control_img = Image.open("common_images/control_bird_canny.png")

controlnet_scales = [0.5]
controlnet_images = [control_img]

# image gem params
negative_prompt = ""
height, width = 1024, 1024
steps = 20
guidance_scale = 7.5
generator = torch.Generator("cuda").manual_seed(123)
num_images = 1

prompts = ["a bird, cartoon style"]
for i in range(len(prompts)):
    for _ in range(3):
        start = time.perf_counter()
        images = model(prompt=prompts[i],
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    # ip_adapter_image_embeds=ip_image_embedding,
                    image=controlnet_images,
                    controlnet_conditioning_scale=controlnet_scales
                    )[0]
        print("cur cost: ", time.perf_counter() - start)
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sdxl_controlnet_{i}.png")
GPUtil.showUtilization(all=True)
