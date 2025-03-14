import os
import sys
import math
from typing import List
from time import time
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms

from s3diff.s3diff_tile import S3Diff
from s3diff.de_net import DEResNet
from s3diff.my_utils.wavelet_color import wavelet_color_fix, adain_color_fix

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

is_old = 1==11

def parse_args_paired_testing(input_args=None):
#    """
#    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
#    This function sets up an argument parser to handle various training options.
#
#    Returns:
#    argparse.Namespace: The parsed command-line arguments.
#   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default=None,)
    parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--de_net_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=96, help="Chopping stride.")
    parser.add_argument("--padding_offset", type=int, default=32, help="padding offset.")

    # parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    # parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    parser.add_argument("--align_method", type=str, default="wavelet")
    
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    parser.add_argument("--neg_prompt", type=str, default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # training details
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    if input_args is not None:
        #args = parser.parse_args(input_args)
        args, unknown = parser.parse_known_args(input_args)
    else:
        #args = parser.parse_args()
        args, unknown = parser.parse_known_args()
    
    return args


class InferenceS3():
    def __init__(self, pretrained_path, 
                        sd_path,
                        de_net_path,
                        vae_encoder_tiled_size,
                        vae_decoder_tiled_size,
                        device='cuda:0', weight_dtype=torch.float32):
        self.pretrained_path = pretrained_path
        self.sd_path = sd_path
        self.de_net_path = de_net_path
        self.vae_encoder_tiled_size = vae_encoder_tiled_size
        self.vae_decoder_tiled_size = vae_decoder_tiled_size
        self.args = parse_args_paired_testing()
        # print(self.args)

        # init sr network
        self.net_sr = S3Diff(lora_rank_unet=self.args.lora_rank_unet, lora_rank_vae=self.args.lora_rank_vae, 
                                sd_path=self.sd_path, pretrained_path=self.pretrained_path, vae_decoder_tiled_size = self.vae_decoder_tiled_size, vae_encoder_tiled_size = self.vae_encoder_tiled_size, args=self.args)
        self.net_sr.set_eval()
        self.net_sr.to(device, dtype=weight_dtype)

        # init degradation estimation network
        self.net_de = DEResNet(num_in_ch=3, num_degradation=2)
        self.net_de.load_model(self.de_net_path)
        self.net_de.eval()
        self.net_de = self.net_de.to(device, dtype=weight_dtype)
        
        print(f"[Inference]: finish initing S3 model")


    @torch.no_grad()
    def process(self, 
        # input_img_path: str,
        input_image: Image.Image,
        scale_factor: float,
        cfg_scale: float,
        latent_tiled_size: int,
        latent_tiled_overlap: int,
        align_method: str,
        device: str,
        ) -> List[np.ndarray]:

        self.net_sr._set_latent_tile(latent_tiled_size = latent_tiled_size, latent_tiled_overlap = latent_tiled_overlap)

        # input_image = Image.open(input_img_path).convert("RGB")
        
        im_lr = tensor_transforms(input_image).unsqueeze(0).to(device)
        ori_h, ori_w = im_lr.shape[2:]
        im_lr_resize = F.interpolate(
            im_lr,
            size=(int(ori_h * scale_factor),
                int(ori_w * scale_factor)),
            mode='bilinear',
            align_corners=True
            )
        im_lr_resize = im_lr_resize.contiguous() 
        im_lr_resize_norm = im_lr_resize * 2 - 1.0
        im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]

        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
        

        with torch.autocast("cuda"):
            deg_score = self.net_de(im_lr)

            pos_tag_prompt = [self.args.pos_prompt]
            neg_tag_prompt = [self.args.neg_prompt]

            x_tgt_pred = self.net_sr(im_lr_resize_norm, deg_score, pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt, 
                            cfg=cfg_scale)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            if is_old:
                out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()
            else:
                out_img = (x_tgt_pred * 0.5 + 0.5)

        if is_old:
            output_pil = transforms.ToPILImage()(out_img[0])

        if align_method == 'no fix':
            image = output_pil
        else:
            if is_old:
                im_lr_resize = transforms.ToPILImage()(im_lr_resize[0].cpu().detach())
            if align_method == 'wavelet':
                if not is_old:
                    image = wavelet_color_fix(out_img[0], im_lr_resize[0])
                else:
                    image = wavelet_color_fix(output_pil, im_lr_resize)
            elif align_method == 'adain':
                image = adain_color_fix(output_pil, im_lr_resize)


        return image


class ProcessS3():
    def __init__(self, pretrained_path, sd_path, de_net_path, device='cuda:0', weight_dtype=torch.float32, vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024, logger=None, **kwargs):
        self.S3 = InferenceS3(pretrained_path=pretrained_path, sd_path=sd_path, de_net_path=de_net_path,device=device,vae_decoder_tiled_size = vae_decoder_tiled_size, vae_encoder_tiled_size = vae_encoder_tiled_size)
        self.logger = logger

    def __call__(self, img_id, lq, sr_scale=2, cfg_scale=1.07, latent_tiled_size=96, latent_tiled_overlap=32, align_method='wavelet', device='cuda:0'):
        try:
            pred = self.S3.process(lq, scale_factor=sr_scale, cfg_scale=cfg_scale,
                                latent_tiled_size=latent_tiled_size, latent_tiled_overlap=latent_tiled_overlap,
                                align_method=align_method, device=device)
            return pred
        except Exception as e:
            error_msg = traceback.format_exc()
            self.logger.error(f"Image {img_id} | s3diff error : {error_msg}")
            return None

if __name__ == "__main__":
    args = {
        'pretrained_path': '/app/weights/s3diff/s3diff.pkl',
        'sd_path': '/app/weights/sd-turbo/',
        'de_net_path': '/app/weights/s3diff/de_net.pth',
        'vae_encoder_tiled_size': 1024,
        'vae_decoder_tiled_size': 224
    }
    S3 = ProcessS3(**args)

    print("ProcessS3 init ok")
    img_path = 'assets/pic/ori0.jpg'
    img_path = 'assets/pic/ori1.jpg'
    img_path = 'assets/pic/ori2.jpg'
    # img_path = '1024_1024.png'
    # img_path = '1920_1280.jpg'
    # img_path = '720_1280.jpg'
    # img_path = '512_512.jpeg'
    input_image = Image.open(img_path).convert('RGB')
    print("image shape: ", input_image.height, input_image.width)
    params = {
        "img_id": 0,
        "lq": input_image,
        "sr_scale": 4,
        "cfg_scale": 1.07,
        "latent_tiled_size": 96,
        "latent_tiled_overlap": 32,
        "align_method": 'wavelet', 
        "device": 'cuda:0'
    }
    # warmup
    image = S3(**params)

    # benchmark
    torch.cuda.synchronize()
    beg = time()
    runcnt = 3
    for i in range(runcnt):
        image = S3(**params)
    torch.cuda.synchronize()
    usetime = time()-beg

    print("average generation time: ", usetime/runcnt)

    image.save("test/res.png")
