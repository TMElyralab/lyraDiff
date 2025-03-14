# Introduction

This [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) demo includes code and scripts for (i) acceleration of basic image generation pipeline and other plugin pipelines using lyraDiff, and (ii) quantizing original models into lyraDiff suitable format


## Ability Matrix
| Syntax                               | Supported |
|:-------------------------------------|:---------:|
| Basic Image Gen                      |    Yes    |
| High Speed LoRA Switch               |    Yes    |
| ControlNet                           |    Yes    |
| FP8 Inference for Basic Image Gen    |    Yes    |
| LoRA for FP8 With Small Quality Loss |    Yes    |
| Int4(SvdQuant) Inference for Basic Image Gen | Experimental |


## Performance on L20 48G

| Height | Width | Batch size | Steps | LoRA num | ControlNet num | Framework    | Time Cost (s) | speed up | GMem (MB) | gmem vs torch (%) |
|:-------:|:------:|:-----------:|:------:|:---------:|:---------------:|:-------------:|:--------------:|:-----------------------:|:----------:|:------------------:|
|   1024  |  1024  |      1     |   20   |     0    |        0        |  torch2.1    |    18.46       |           1.0x           |   37078    |        0.0         |
|   1024  |  1024  |      1     |   20   |     0    |        0        |  lyraDiff    |    15.07       |          1.2x           |   37480    |       -1.1         |
|   1024  |  1024  |      1     |   20   |     0    |        0        | lyraDiff-fp8 |    10.23       |          1.8x           |   23614    |       36.3         |
|   1024  |  1024  |      1     |   20   |     0    |        0        | lyraDiff-int4|    14.68       |           1.25x           |   19902    |       46.9         |
|   1024  |  1024  |      1     |   20   |     1    |        0        |  torch2.1    |    18.46       |          1.0x          |   37078    |        0.0         |
|   1024  |  1024  |      1     |   20   |     1    |        0        |  lyraDiff    |    15.07       |          1.2x           |   37480    |       -1.1         |
|   1024  |  1024  |      1     |   20   |     1    |        0        | lyraDiff-fp8 |    10.23       |          1.8x           |   23614    |       36.3         |
|   1024  |  1024  |      1     |   20   |     3    |        0        |  torch2.1    |    18.46       |         1.0x          |   37078    |        0.0         |
|   1024  |  1024  |      1     |   20   |     3    |        0        |  lyraDiff    |    15.07       |          1.2x            |   37480    |       -1.1         |
|   1024  |  1024  |      1     |   20   |     3    |        0        | lyraDiff-fp8 |    10.23       |          1.8x            |   23614    |       36.3         |
|   1024  |  1024  |      1     |   20   |     0    |        1        |  torch2.1    |    19.27       |           1.0x           |   38736    |        0.0         |
|   1024  |  1024  |      1     |   20   |     0    |        1        |  lyraDiff    |    15.48       |          1.2x            |   39742    |       -2.6         |
|   1024  |  1024  |      1     |   20   |     0    |        1        | lyraDiff-fp8 |    10.66       |          1.8x            |   24614    |       36.5         |

## Model Select
It is highly recommended to use our converted version of the [FLUX.1-dev](https://huggingface.co/TMElyralab/lyraDiff-Flux.1-dev) model. You can use it in both non-quant (bf16) and fp8 inference.

```bash
huggingface-cli download TMElyralab/lyraDiff-Flux.1-dev --locallocal-dir </path/to/model>

```

## Usage Guide
We provide two ways to use the lyraDiff version of FLUX transformer model for image generation:
1. Using `LyraDiffFluxTransformer2DModel`, the entire model object is implemented in C++/Cuda, which provides better performance.
2. Using `LyraDiffFluxTransformer2DModelV2`, only the `SingleTransformerBlock` and `TransformerBlock` objects are implemented in C++/Cuda, which provides better flexibility.

We here use `LyraDiffFluxTransformer2DModelV2` for example, but all apis of these 2 are the same

### Basic Image gen
Refer to [flux_demo.py](flux_demo.py)
``` python

import sys
sys.path.append("../..")

import torch
import time
from diffusers import FluxPipeline, AutoencoderKL
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
from lyradiff.lyradiff_model.lyradiff_vae_model import LyraDiffVaeModel
from lyradiff.lyradiff_model.lora_util import flux_load_lora, flux_clear_lora
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel
from lyradiff.lyradiff_model.module.lyradiff_ip_adapter import LyraIPAdapter
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from PIL import Image
from optimum.quanto import freeze, qfloat8, quantize, QuantizedTransformersModel

model_path = "/path/to/lyraDiff-FLUX.1-dev/"

# Here we provide 3 difference quant_level choice, and the LyraQuantLevel.FP8_W8A8_FULL is highly recommended
# quant_level = LyraQuantLevel.INT4_W4A4_FULL
quant_level = LyraQuantLevel.FP8_W8A8_FULL
# quant_level = LyraQuantLevel.NONE
transformer_model = LyraDiffFluxTransformer2DModelV2(quant_level=quant_level)
start = time.perf_counter()
transformer_model.load_from_diffusers_model(os.path.join(model_path, "transformer"))

print(f"after load transformer_model: {time.perf_counter() - start}")
GPUtil.showUtilization(all=True)

model = FluxPipeline.from_pretrained(
    model_path,
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
model.transformer = transformer_model

prompt = "Female furry Pixie with text hello world"

# Image Gen
generator=torch.Generator("cuda").manual_seed(123)

for i in range(3):
    # generator = torch.Generator("cuda").manual_seed(123)
    start = time.perf_counter()
    images = model(prompt,
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
```

### Dynamically Switch LoRA
Refer to [flux_demo.py](flux_demo.py)
``` python
from lyradiff.lyradiff_model.lora_util import flux_load_lora, flux_clear_lora

lora_file_list1 = ["/path/to/flux/lora.safetensors"]
lora_alpha_list1 = [0.5]

lora_state_dict_list = flux_load_lora(lora_file_list1, lora_alpha_list1, transformer_model, model.text_encoder, model.text_encoder_2, quant_level)

```

### Controlnet Support
Refer to [flux_controlnet_demo.py](flux_controlnet_demo.py)
```python
import sys
sys.path.append("../..")

import torch
import time
from diffusers import FluxPipeline, AutoencoderKL, FluxControlNetModel, FluxControlNetPipeline
import GPUtil
import os
from glob import glob
import random
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model_v2 import LyraDiffFluxTransformer2DModelV2
from lyradiff.lyradiff_model.lyradiff_flux_controlnet_model import LyraDiffFluxControlNetModel
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from PIL import Image
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

# image gen
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

```

## Model Quantization

### FLUX Model Quantize FP8

#### Step 1: Quantize Model
We leverage the quantize method of [Nvidia TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/quantization)
[quantize.py](../../lyradiff/convert_model_scripts/quantize.py) script to convert flux.dev model to fp8 version:

##### Params
- `model`: model select choice from [flux, sd1.5, sdxl]
- `model-path`: input model path
- `quantized-torch-ckpt-save-path`: output path
- `format`: fp8/int8 to quantize fp8 or int8 model
- `n_steps`: calibrate steps, default=30
- `batch-size`: calibrate batch size, default=1
- `calib-size`: calibrate prompt size, SDXL recommend 64, SD1.5 recommend 512, flux recommend 128
- `alpha`: only used for int8 smoothquant, SDXL recommend 0.8, SD1.5 recommend 1.0
- `quant-level`: quant level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, we only support `3` for now

Example:
``` bash
python3 quantize.py --model flux-dev --model-path <path/to/diffusers_model/> --quantized-torch-ckpt-save-path model_fp8.pt --quant-level 3 --format fp8 --calib-size 128  --n-steps 20
```
#### Step 2: Convert quantzied
Use [convert_flux_fp8_safetensors.py](../../lyradiff/convert_model_scripts/convert_flux_fp8_safetensors.py) script to convert the generated `model_fp8.pt` to diffusers safetensors format so that quant and non-quant mode can share same model

##### Params
- `quantized-model-path`: quantized model_fp8.pt model path
- `optimized-model-dir`: output model path
- `--fp16`: whether to save model in fp16

Example:
``` bash
python3 convert_flux_fp8_safetensors.py --quantized-model-path  --model-path <path/to/diffusers_model/> --quantized-torch-ckpt-save-path model_fp8.pt --quant-level 3 --format fp8 --calib-size 128  --n-steps 20
```