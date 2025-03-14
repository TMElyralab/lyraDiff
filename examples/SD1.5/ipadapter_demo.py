import torch
import time
from lyradiff.lyradiff_model import LyraDiffUNet2DConditionModel, LyraDiffVaeModel
from diffusers import StableDiffusionPipeline
from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter
import GPUtil
import os
from PIL import Image

model_path = "/path/to/sd1.5/model"

unet = LyraDiffUNet2DConditionModel(is_sdxl=False)
vae = LyraDiffVaeModel()

unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(os.path.join(model_path, "vae"))

pipe = StableDiffusionPipeline.from_pretrained(model_path).to(dtype=torch.float16).to("cuda")
pipe.unet = unet
pipe.vae = vae

ip_ckpt = "/path/to/sd/ip_ckpt/ip-adapter_sd15.bin"
image_encoder_path = "/path/to/sd/ip_ckpt/image_encoder"

# Create LyraIPAdapter
ip_adapter = LyraIPAdapter(unet_model=unet.model, sdxl=False, device=torch.device("cuda"), ip_ckpt=ip_ckpt, ip_plus=False, image_encoder_path=image_encoder_path)

# load ip_adapter image
ip_image = Image.open(
    "sd1_5_0.png")
ip_scale = 0.5

# get ip image embedding and pass it to the pipeline
ip_image_embedding = [ip_adapter.get_image_embeds_lyradiff(ip_image)['ip_hidden_states']]
# unet set ip adapter scale in unet model obj, since we cannot set ip_adapter_scale through diffusers pipeline
unet.set_ip_adapter_scale(ip_scale)

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
                   ip_adapter_image_embeds=ip_image_embedding
                   )[0]
    print("cur cost: ", time.perf_counter() - start)
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sd1_5_ip_{i}.png")
GPUtil.showUtilization(all=True)
