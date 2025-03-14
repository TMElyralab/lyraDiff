# Introduction

This [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) demo includes code and scripts for (i) acceleration of basic image generation pipeline and other plugin pipelines using lyraDiff, and (ii) converting or quantizing original models into lyraDiff suitable format


## Ability Matrix
| Ability | Supported |
|:----------- |:-----------:|
| Basic Image Gen | Yes |
| High Speed LoRA Swap | Yes |
| ControlNet | Yes |
| Ip-Adapter | Yes |

## Performance

| Height | Width | Batch size | Steps | LoRA num | ControlNet num | Framework    | Time cost (s) | speed up | GMem (MB) | gmem vs torch |
|:-------:|:------:|:-----------:|:------:|:---------:|:---------------:|:-------------:|:---------------:|:--------------------:|:----------:|:---------------:|
| 1024   | 1024  | 1          | 20    | 0        | 0              | torch2.1  | 2.93          | 1.0x               | 8973      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 0        | 0              | xformers  | 2.11          | 1.4x              | 8973      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 0        | 0              | lyraDiff  | 1.40          | 2.1x              | 6045      | 32.6%         |
| 1024   | 1024  | 1          | 20    | 1        | 0              | torch2.1  | 2.93          | 1.0x               | 9241      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 1        | 0              | xformers  | 2.11          | 1.4x              | 9261      | -0.2%         |
| 1024   | 1024  | 1          | 20    | 1        | 0              | lyraDiff  | 1.40          | 2.1x              | 6409      | 30.7%         |
| 1024   | 1024  | 1          | 20    | 3        | 0              | torch2.1  | 2.93          | 1.0x               | 9265      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 3        | 0              | xformers  | 2.11          | 1.4x              | 9265      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 3        | 0              | lyraDiff  | 1.40          | 2.1x              | 6409      | 30.8%         |
| 1024   | 1024  | 1          | 20    | 0        | 1              | torch2.1  | 4.11          | 1.0x               | 9733      | 0.0%          |
| 1024   | 1024  | 1          | 20    | 0        | 1              | xformers  | 2.96          | 1.4x              | 9743      | -0.1%         |
| 1024   | 1024  | 1          | 20    | 0        | 1              | lyraDiff  | 1.94          | 2.1x              | 6864      | 29.5%         |
| 1024   | 1024  | 1          | 20    | 0        | 3              | torch2.1  | 6.42          | 1.0x               | 11147     | 0.0%          |
| 1024   | 1024  | 1          | 20    | 0        | 3              | xformers  | 4.61          | 1.4x               | 11165     | -0.2%         |
| 1024   | 1024  | 1          | 20    | 0        | 3              | lyraDiff  | 3.04          | 2.1x              | 8833      | 20.8%         |

## Model Select
You can basically use whatever sd1.5 base, LoRA and ControlNet models you want as long as it's in diffusers model format

## Usage Guide

### Basic Image gen
Refer to [txt2img_demo.py](txt2img_demo.py)
``` python

import sys
sys.path.append("../..")

import torch
import time
from diffusers import StableDiffusionPipeline
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPTextModel, CLIPTokenizer
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from PIL import Image

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


```

### Dynamically Switch LoRA
Refer to [txt2img_demo.py](txt2img_demo.py)
``` python
from lyradiff.lyradiff_model.lora_util import load_lora_state_dict, add_lora_to_opt_model

lora_scale = 1.0

lora_state = load_lora_state_dict("/path/to/sd/lora.safetensors")
add_lora_to_opt_model(lora_state, lyra_unet, clip_model, None, alpha=1.0)
```

### Controlnet Support
Refer to [controlnet_demo.py](controlnet_demo.py)
``` python
import torch
import time
from diffusers import StableDiffusionControlNetPipeline
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from lyradiff.lyradiff_model.lyradiff_controlnet_model import LyraDiffControlNetModel
from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
from diffusers.utils import load_image

model_path = "/path/to/sd/model/"
controlnet_model_path = "/path/to/sd/controlnet/model/"

text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder").to(torch.float16).to(torch.device("cuda"))

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer")

# init lyradiff version of unet, vae and controlnet
unet = LyraDiffUNet2DConditionModel(is_sdxl=False)
vae = LyraDiffVaeModel()
controlnet = LyraDiffControlNetModel(unet)

# load state dict from diffusers format models
unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(os.path.join(model_path, "vae"))
controlnet.load_from_diffusers_model(
    model_name="canny", controlnet_path=controlnet_model_path)
controlnet_list = MultiControlNetModel([controlnet])

scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_path, subfolder="scheduler", timestep_spacing="linspace")

pipe = StableDiffusionControlNetPipeline(
    vae=vae,
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    controlnet=controlnet_list
).to(torch.float16).to(torch.device("cuda"))

control_img = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/bird.png")

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
# img gen
for i in range(len(prompts)):
    for _ in range(3):
        start = time.perf_counter()
        images = pipe(prompt=prompts[i],
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
    images[0].save(f"sd_controlnet_{i}.png")
GPUtil.showUtilization(all=True)
```

### IpAdapter Support
Refer to [ipadapter_demo.py](ipadapter_demo.py)
``` python
from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter

ip_ckpt = "/path/to/sd/ip_ckpt/ip-adapter_sd15-h.bin"
image_encoder_path = "/path/to/sd/ip_ckpt/image_encoder"

# Create LyraIPAdapter
ip_adapter = LyraIPAdapter(unet_model=unet.model, sdxl=False, device=torch.device("cuda"), ip_ckpt=ip_ckpt, ip_plus=False, image_encoder_path=image_encoder_path)

# load ip_adapter image
ip_image = Image.open(
    "ip_image.png")
ip_scale = 0.5

# get ip image embedding and pass it to the pipeline
ip_image_embedding = [ip_adapter.get_image_embeds_lyradiff(ip_image)['ip_hidden_states']]

# set ip adapter scale in unet model obj, since we cannot set ip_adapter_scale through diffusers pipeline
unet.set_ip_adapter_scale(ip_scale)
# img gen
for i in range(len(prompts)):
    for _ in range(3):
        start = time.perf_counter()
        images = pipe(prompt=prompts[i],
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    ip_adapter_image_embeds=ip_image_embedding,
                    )[0]
        print("cur cost: ", time.perf_counter() - start)
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sd_ip_{i}.png")
GPUtil.showUtilization(all=True)

```

__NOTE__: We need to use a converted version for [IP-Adapter](https://huggingface.co/TMElyralab/lyraDiff-IP-Adapters) models, you can either download our pre-converted version or convert it yourself [Model Convert](#Model Convert).

To download lyraDiff format of IP-Adapter Models, please try this
```bash
huggingface-cli download TMElyralab/lyraDiff-IP-Adapters --locallocal-dir </path/to/ip-model>

# Use lyraDiff-IP-Adapters/lyra_tran/models/ip-adapter_sd15.bin for ip-ckpt
# Use lyraDiff-IP-Adapters/lyra_tran/models/image_encoder for image_encoder

```

## Model Convert

### IP-Adapter Model Convert

Use [convert_ipadapter.py](../../lyradiff/convert_model_scripts/convert_ipadapter.py) script to do IP-Adapter model convert:

#### Params
- `model_dir`: IP model dir 
- `subfolder`: subfolder of IP model ckpt
- `weight_name`: ip ckpt name, usually `ip-adapter_sd15.bin` for SD1.5 and `ip-adapter-plus_sdxl_vit-h.bin` for SDXL 
- `safetensors`: if model weight is in safetensors format
- `is_xl`: if converting SDXL IP-Adapter model

Example:
``` bash
python3 convert_ipadapter.py --model_dir=</path/to/ip-model> --subfolder=models --weight_name=ip-adapter_sd15.bin
```
Converted model will be placed under ${model_dir}/lyra_tran/${subfolder}/${weight_name}
