import torch
import time
from lyradiff.lyradiff_model import LyraDiffUNet2DConditionModel, LyraDiffVaeModel
from diffusers import StableDiffusionPipeline
import GPUtil
import os

model_path = "/path/to/sd1.5/model"

unet = LyraDiffUNet2DConditionModel(is_sdxl=False)
vae = LyraDiffVaeModel()

unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(os.path.join(model_path, "vae"))

pipe = StableDiffusionPipeline.from_pretrained(model_path).to(dtype=torch.float16).to("cuda")
pipe.unet = unet
pipe.vae = vae

# img gen params
prompts = ["a beautiful girl, cartoon style"]
negative_prompt = "NSFW"
height, width = 512, 512
steps = 20
guidance_scale = 7.5
generator = torch.Generator("cuda").manual_seed(123)
num_images = 1

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
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sd1_5_{i}.png")
GPUtil.showUtilization(all=True)
