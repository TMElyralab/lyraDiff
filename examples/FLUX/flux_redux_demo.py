import torch
from diffusers import FluxPipeline, AutoencoderKL, FluxPriorReduxPipeline
from diffusers.utils import load_image
import GPUtil
import os
from transformers import  CLIPTokenizer,  T5TokenizerFast
from lyradiff.lyradiff_model.lyradiff_utils import LyraQuantLevel
from lyradiff.lyradiff_model.lyradiff_flux_transformer_model import LyraDiffFluxTransformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler

model_path = "/path/to/LyraDiff-FLUX.1-dev/"

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer")
tokenizer_2 = T5TokenizerFast.from_pretrained(
    model_path, subfolder="tokenizer_2")

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch.bfloat16).to(torch.device("cuda"))

quant_level = LyraQuantLevel.NONE
transformer_model = LyraDiffFluxTransformer2DModel(quant_level=quant_level)
transformer_model.load_from_diffusers_model(os.path.join(model_path, "transformer"))

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_path, subfolder="scheduler")

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

model = FluxPipeline(
    vae=vae,
    transformer=transformer_model,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler
)

prompts = ["Female furry Pixie with text hello world", "Female furry Pixie with text hello world", "Female furry Pixie with text hello world"]
# prompts = ["A cat holding a sign that says hello world", "A cat holding a sign that says hello world", "A cat holding a sign that says hello world"]

# image gen
generator=torch.Generator("cuda").manual_seed(123)

pipe_prior_output = pipe_prior_redux(image)
images = model(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(0),
    **pipe_prior_output,
).images
images[0].save("lyradiff-flux-dev-redux.png")
    
GPUtil.showUtilization(all=True)
