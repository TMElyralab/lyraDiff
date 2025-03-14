# Introduction

This [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) demo includes code and scripts for (i) acceleration of basic image generation pipeline and other plugin pipelines using lyraDiff, and (ii) converting or quantizing original models into lyraDiff suitable format

## Ability Matrix
| Ability | Supported |
|:----------- |:-----------:|
| Basic Image Gen | Yes |
| High Speed LoRA Switch | Yes |
| Controlnet | Yes |
| IP-adapter  | Yes |
| SmoothQuant (int8) for Basic Image Gen | Yes |

## Performance on A100 40G

| Height | Width | Batch size | Steps | LoRA num | ControlNet num | Framework    | Time Cost (s) | speed up | GMem (MB) | gmem vs torch (%) |
|:-------------------------------------------------:|:------:|:-----------:|:------:|:---------:|:---------------:|:-------------:|:--------------:|:-----------------------:|:----------:|:------------------:|
| 1024                                             | 1024  | 1          | 20    | 0        | 0              | torch2.1     | 2.60          | 1.0x                    | 16309     | 0.0               |
| 1024                                             | 1024  | 1          | 20    | 0        | 0              | xformers     | 2.51          | 1.0x                    | 16363     | -0.3              |
| 1024                                             | 1024  | 1          | 20    | 0        | 0              | lyraDiff     | 1.73          | 1.5x                   | 15437     | 5.4               |
| 1024                                             | 1024  | 1          | 20    | 0        | 0              | lyraDiff-int8| 1.49          | 1.75x                   | 13607     | 16.8              |
| 1024                                             | 1024  | 1          | 20    | 1        | 0              | torch2.1     | 2.61          |  1.0x                   | 17613     | 0.0               |
| 1024                                             | 1024  | 1          | 20    | 1        | 0              | xformers     | 2.58          | 1.0x                    | 17611     | 0.1               |
| 1024                                             | 1024  | 1          | 20    | 1        | 0              | lyraDiff     | 1.73          | 1.5x                   | 16761     | 4.8               |
| 1024                                             | 1024  | 1          | 20    | 3        | 0              | torch2.1     | 2.61          | 1.0x                   | 17689     | 0.0               |
| 1024                                             | 1024  | 1          | 20    | 3        | 0              | xformers     | 2.53          | 1.0x                   | 17685     | 0.2               |
| 1024                                             | 1024  | 1          | 20    | 3        | 0              | lyraDiff     | 1.73          | 1.5x                   | 16761     | 5.2               |
| 1024                                             | 1024  | 1          | 20    | 0        | 1              | torch2.1     | 3.73          | 1.0x                     | 18867     | 0.0               |
| 1024                                             | 1024  | 1          | 20    | 0        | 1              | xformers     | 3.55          | 1.0x                     | 18799     | 0.4               |
| 1024                                             | 1024  | 1          | 20    | 0        | 1              | lyraDiff     | 2.49          | 1.5x                    | 18539     | 1.7               |
| 1024                                             | 1024  | 1          | 20    | 0        | 3              | torch2.1     | 5.96          | 1.0x                    | 23689     | 0.0               |
| 1024                                             | 1024  | 1          | 20    | 0        | 3              | xformers     | 5.70          | 1.0x                    | 23805     | -0.5              |
| 1024                                             | 1024  | 1          | 20    | 0        | 3              | lyraDiff     | 4.06          | 1.5x                    | 23791     | -0.4              |

## Model Select
You can use any SDXL base, LoRA, and ControlNet (Large and Small) models as long as they are in the diffusers model format. It is highly recommended to use the [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) VAE instead of the default one, as it can lower the DRAM usage.

## Usage Guide
We provide two ways to use lyraDiff for image generation:
1. Using our version of image generation pipelines, which offers better performance but less flexibility.
2. Using the diffusers' original image generation pipelines, which offers better flexibility but less performance.

### Basic Image gen
Refer to [txt2img_demo.py](txt2img_demo.py)
``` python
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


model_path = "/path/to/sdxl/model"
vae_model_path = "/path/to/sdxl/vae/model/sdxl-vae-fp16-fix"

text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder").to(torch.float16).to(torch.device("cuda"))

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_path, subfolder="text_encoder_2").to(torch.float16).to(torch.device("cuda"))

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer_2")

# init lyradiff version of unet and vae
unet = LyraDiffUNet2DConditionModel(is_sdxl=True)
vae = LyraDiffVaeModel(scaling_factor=0.13025, is_upcast=False)

unet.load_from_diffusers_model(os.path.join(model_path, "unet"))
vae.load_from_diffusers_model(vae_model_path)

# init scheduler
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
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sdxl_{i}.png")

```

### Basic Image gen with INT8
``` python
model_path = "/path/to/sdxl/model"

quant_level=LyraQuantLevel.INT8_SMOOTH_QUANT_LEVEL3

# init model in quant mode
unet = LyraDiffUNet2DConditionModel(is_sdxl=True, quant_level=quant_level)

# load model from int8 version
unet.load_config(os.path.join(model_path, "unet/config.json"))
unet.load_from_bin(os.path.join(model_path, "unet_bins_fp16"))

```

### Dynamically Switch LoRA
Refer to [txt2img_demo.py](txt2img_demo.py)
``` python
from lyradiff.lyradiff_model.lora_util import load_lora_state_dict, add_lora_to_opt_model

lora_scale = 1.0

lora_state = load_lora_state_dict("/path/to/sdxl/lora.safetensors")
add_lora_to_opt_model(lora_state, lyra_unet, clip_model, clip_model_2, alpha=1.0)
```

### Controlnet Support
Refer to [controlnet_demo.py](controlnet_demo.py)
``` python
import torch
import time
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
from lyradiff.lyradiff_model.lyradiff_unet_model import LyraDiffUNet2DConditionModel
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from lyradiff.lyradiff_model.lyradiff_controlnet_model import LyraDiffControlNetModel
from lyradiff.lyradiff_model.lora_util import load_lora_state_dict, add_lora_to_opt_model
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
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

```

### IP-Adapter Support

Refer to [ipadapter_demo.py](ipadapter_demo.py)
``` python

from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter

ip_ckpt = "/path/to/sdxl/ip_ckpt/ip-adapter-plus_sdxl_vit-h.bin"
image_encoder_path = "/path/to/sdxl/ip_ckpt/image_encoder"

# Create LyraIPAdapter
ip_adapter = LyraIPAdapter(unet_model=unet.model, sdxl=True, device=torch.device(
    "cuda"), ip_ckpt=ip_ckpt, ip_plus=True, image_encoder_path=image_encoder_path, num_ip_tokens=16, ip_projection_dim=1024)

# load ip_adapter image
ip_image = Image.open(
    "ip_image.png")
ip_scale = 0.5

# get ip image embedding and pass it to the pipeline
ip_image_embedding = [ip_adapter.get_image_embeds_lyradiff(ip_image)['ip_hidden_states']]
# unet set ip adapter scale in unet model obj, since we cannot set ip_adapter_scale through diffusers pipeline
unet.set_ip_adapter_scale(ip_scale)

# img gen
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
                    ip_adapter_image_embeds=ip_image_embedding,
                    )[0]
        print("cur cost: ", time.perf_counter() - start)
    # images[0].save(f"outputs/{lora_name}-{i}.png")
    images[0].save(f"sdxl_ip_{i}.png")
GPUtil.showUtilization(all=True)

```

__NOTE__: We need to use a converted version for [IP-Adapter](https://huggingface.co/TMElyralab/lyraDiff-IP-Adapters) models, you can either download our pre-converted version or convert it yourself [Model Convert](#Model Convert).

To download lyraDiff format of IP-adapter Models, please try this:
```bash
huggingface-cli download TMElyralab/lyraDiff-IP-Adapters --locallocal-dir </path/to/ip-model>

# Use lyraDiff-IP-Adapters/lyra_tran/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin for ip-ckpt
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
python3 convert_ipadapter.py --model_dir=</path/to/ip-model> --subfolder=models --weight_name=ip-adapter-plus_sdxl_vit-h.bin --is_xl
```
Converted model will be placed under ${model_dir}/lyra_tran/${subfolder}/${weight_name}

### SDXL Model Quantization Int8 

Use [convert_sd_quant_model.py](../../lyradiff/convert_model_scripts/convert_sd_quant_model.py) to quantize sdxl model to int8 version:

#### Params
- `sd-ckpt-file`: model path
- `optimized-model-dir`: output path
- `fp16`: fp8/int8 to quantize fp8 or int8 model
- `use_safetensors`: fp8/int8 to quantize fp8 or int8 model
- `sdxl`: if converting sdxl model
- `n_steps`: calibrate steps, default=30
- `collect-method`: not used for now
- `batch-size`: calibrate batch size, default=1
- `calib-size`: calibrate prompt size, SDXL recommend 64, SD1.5 recommend 512, flux recommend 128
- `alpha`: only used for int8 smoothquant, SDXL recommend 0.8, SD1.5 recommend 1.0
- `quant-level`: quant level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, we only support `3` for now


Example:
``` bash
python3 convert_sd_quant_model.py  --sd-ckpt-file  /path/to/sdxl/model/ --optimized-model-dir ./output --sdxl --use_safetensors  --fp16  --batch-size 1 --calib-size 1024  --alpha 1.0
```
Converted model will be placed under `<optimized-model-dir>` in diffusers style, and int8 version of unet weights are save in `<optimized-model-dir>/unet_bins_fp16`
