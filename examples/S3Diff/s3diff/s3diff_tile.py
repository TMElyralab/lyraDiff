import os
import re
import requests
import sys
import time
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
# p = "src/"
# sys.path.append(p)
sys.path.append(os.getcwd()+'/s3diff')
from model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
from my_utils.vaehook import VAEHook, perfcount
from lyra_utils.lyra_s3diff_utils import *
from lyra_utils.lyra_vae_model import LyraDiffVaeModel
import my_utils.devices as devices


is_old = 1==11

def timedot0(name, t):
    print("duration: ", name, time.perf_counter()-t)

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64, vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024, args=None):
        super().__init__()
        self.is_lyra_unet = 1==1
        self.is_lyra_vae  = 1==1
        if is_old:
            self.is_lyra_unet = False
            self.is_lyra_vae = False
        11==11
        self.args = args
        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)
        self.guidance_scale = 1.07

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            if not self.is_lyra_vae:
                vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                _sd_vae = vae.state_dict()
                for k in sd["state_dict_vae"]:
                    _sd_vae[k] = sd["state_dict_vae"][k]
                vae.load_state_dict(_sd_vae)

            if not self.is_lyra_unet:
                unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
                unet.add_adapter(unet_lora_config)
                _sd_unet = unet.state_dict()
                for k in sd["state_dict_unet"]:
                    _sd_unet[k] = sd["state_dict_unet"][k]
                unet.load_state_dict(_sd_unet)

            _vae_de_mlp = self.vae_de_mlp.state_dict()
            for k in sd["state_dict_vae_de_mlp"]:
                _vae_de_mlp[k] = sd["state_dict_vae_de_mlp"][k]
            self.vae_de_mlp.load_state_dict(_vae_de_mlp)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _vae_block_mlp = self.vae_block_mlp.state_dict()
            for k in sd["state_dict_vae_block_mlp"]:
                _vae_block_mlp[k] = sd["state_dict_vae_block_mlp"][k]
            self.vae_block_mlp.load_state_dict(_vae_block_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
            for k in sd["state_dict_vae_fuse_mlp"]:
                _vae_fuse_mlp[k] = sd["state_dict_vae_fuse_mlp"][k]
            self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

            self.W = nn.Parameter(sd["w"], requires_grad=False)

            embeddings_state_dict = sd["state_embeddings"]
            self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
            self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # vae tile
        self._init_tiled_vae(encoder_tile_size=vae_encoder_tiled_size, decoder_tile_size=vae_decoder_tiled_size)

        ######### lyra #########
        def update_lyra_unet():
            unet_lyra = torch.classes.lyradiff.Unet2dConditionalModelOp(
                3,
                "fp16",
                4,4,0, sd_ver="sd2"
            )
            unet_new = LyraUnet(unet_lyra, self.unet.config)
            fpath_s3diff = "/app/weights/V2/s3diff.pkl"
            state_dict_s3diff = torch.load(fpath_s3diff)
            sd = state_dict_s3diff
            
            state_dict = load_file("/app/weights/sd-turbo/unet/diffusion_pytorch_model.safetensors", "cuda")
            state_dict_unet = sd["state_dict_unet"]
            state_dict["conv_in.weight"] = state_dict_unet["conv_in.weight"]
            state_dict["conv_in.bias"] = state_dict_unet["conv_in.bias"]
            for key in state_dict:
                if len(state_dict[key].shape) == 4:
                    state_dict[key] = state_dict[key].cuda().to(
                        torch.float16).permute(0, 2, 3, 1).contiguous()
                state_dict[key] = state_dict[key].cuda().to(torch.float16)
            unet_lyra.reload_unet_model_from_cache(state_dict, "cuda")

            dict_lora_A, dict_lora_B, scaling = load_s3diff_lora(state_dict_unet, 32)
            unet_lyra.load_s3diff_lora_from_state_dict(dict_lora_A, dict_lora_B)
            self.dict_lora_A = dict_lora_A
            self.dict_lora_B = dict_lora_B
            self.unet.to("cpu")
            self.unet_lyra = unet_lyra
            del self.unet

            self.unet = unet_new

        def update_lyra_vae():
            torch.cuda.synchronize()
            # torch.classes.load_library("/data/home/kioka/lyradiff/build/lib/libth_lyradiff.so")
            vae_scale_factor=8
            vae_scaling_factor=0.18215
            vae_lyra = LyraDiffVaeModel(
                scale_factor=vae_scale_factor, scaling_factor=vae_scaling_factor, is_upcast=False)

            # load weight
            state_dict = load_file("/app/weights/sd-turbo/vae/diffusion_pytorch_model.fp16.safetensors", "cuda") 
            state_dict = vae_lyra.convert_state_dict(state_dict)
            vae_lyra.model.reload_vae_model_from_cache(state_dict, "cuda")
            vae_lyra.enable_tiling()

            torch.cuda.synchronize()
            state_dict_vae = sd["state_dict_vae"]
            dict_lora_A_vae, dict_lora_B_vae, scaling = load_s3diff_lora(state_dict_vae, 16)
            vae_lyra.model.load_s3diff_lora_from_state_dict(dict_lora_A_vae, dict_lora_B_vae)
            torch.cuda.synchronize()
            self.dict_lora_A_vae = dict_lora_A_vae
            self.dict_lora_B_vae = dict_lora_B_vae

            vae_lyra.load_config(self.vae.config)
            self.vae.to("cpu")
            
            del self.vae
            self.vae = vae_lyra

            torch.cuda.synchronize()
            self.vae_lyra = vae_lyra

            vae_lyra.enable_dynamic_tiling_s3diff()
            vae_lyra.set_tile_pad(vae_encoder_tiled_size, 32, False)
            vae_lyra.set_tile_pad(vae_decoder_tiled_size, 11, True)


        if self.is_lyra_unet:
            update_lyra_unet()

        if self.is_lyra_vae:
            update_lyra_vae()



    def set_eval(self):
        if hasattr(self.unet, "eval"):
            self.unet.eval()
            self.unet.requires_grad_(False)
        if hasattr(self.vae, "eval"):
            self.vae.eval()
            self.vae.requires_grad_(False)

        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp.train()
        self.unet_de_mlp.train()
        self.vae_block_mlp.train()
        self.unet_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.unet_fuse_mlp.train()    

        self.vae_block_embeddings.requires_grad_(True)
        self.unet_block_embeddings.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    @perfcount
    @torch.no_grad()
    def forward(self, c_t, deg_score, pos_prompt, neg_prompt, cfg):
        print("============= s3diff inp shape:", c_t.shape)
        self.guidance_scale = cfg
        beg_time = time.perf_counter()

        if pos_prompt is not None:
            # encode the text prompt
            pos_caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            pos_caption_enc = self.text_encoder(pos_caption_tokens)[0]
        else:
            pos_caption_enc = self.text_encoder(prompt_tokens)[0]

        if neg_prompt is not None:
            # encode the text prompt
            neg_caption_tokens = self.tokenizer(neg_prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            neg_caption_enc = self.text_encoder(neg_caption_tokens)[0]
        else:
            neg_caption_enc = self.text_encoder(neg_prompt_tokens)[0]

        # degradation fourier embedding
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        if not self.is_lyra_vae:
            for layer_name, module in self.vae.named_modules():
                if layer_name in self.vae_lora_layers:
                    split_name = layer_name.split(".")
                    if split_name[1] == 'down_blocks':
                        block_id = int(split_name[2])
                        vae_embed = vae_embeds[:, block_id]
                    elif split_name[1] == 'mid_block':
                        vae_embed = vae_embeds[:, -2]
                    else:
                        vae_embed = vae_embeds[:, -1]
                    module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)
        else:
            map_extra_tensors_vae = parse_unet_embeds(vae_embeds, True)
            scale_params_vae = {"s3diff_lora_scaling": 0.5}

        if not self.is_lyra_unet:
            for layer_name, module in self.unet.named_modules():
                if layer_name in self.unet_lora_layers:
                    split_name = layer_name.split(".")
                    if split_name[0] == 'down_blocks':
                        block_id = int(split_name[1])
                        unet_embed = unet_embeds[:, block_id]
                    elif split_name[0] == 'mid_block':
                        unet_embed = unet_embeds[:, 4]
                    elif split_name[0] == 'up_blocks':
                        block_id = int(split_name[1]) + 5
                        unet_embed = unet_embeds[:, block_id]
                    else:
                        unet_embed = unet_embeds[:, -1]
                    module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)
        else:
            map_extra_tensors = parse_unet_embeds(unet_embeds)
            scale_params = {"s3diff_lora_scaling": 0.25}
            # print("dtype unet_embeds", unet_embeds.dtype)

        timedot0("before vae_encode", beg_time)
        print(">>>================================================================================")
        print("vae encode shape:", c_t.shape)
        if not self.is_lyra_vae:
            lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            lq_latent = self.vae.encode(c_t, map_extra_tensors=map_extra_tensors_vae, scale_params={"s3diff_lora_scaling": 0.5}).latent_dist.sample() * self.vae.config.scaling_factor
        # gt:   vae encode shape torch.Size([1, 4, 288, 160])
        # lyra: vae encode shape torch.Size([1, 4, 288, 160])
        print("vae encode shape", lq_latent.shape)
        print("<<<================================================================================")

        timedot0("after vae_encode", beg_time)
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        t0 = time.perf_counter()
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            if not is_old:
                input_list_t2 = torch.cat([lq_latent, lq_latent], dim=0)
                encoder_hidden_states2 = torch.cat([pos_caption_enc, neg_caption_enc], dim=0).to(torch.float16)
                if self.is_lyra_unet:
                    preds = self.unet(input_list_t2, encoder_hidden_states2, self.timesteps, map_extra_tensors, scale_params)
                else:
                    preds = self.unet(input_list_t2,  self.timesteps, encoder_hidden_states2).sample
                model_pred = preds[1] + self.guidance_scale * (preds[0] - preds[1])
            else:
                pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
                model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")
            # tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to()
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(c_t.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        if is_old:
                            input_list_t = torch.cat(input_list, dim=0)
                            # predict the noise residual
                            pos_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                            neg_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=neg_caption_enc).sample
                            model_out = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
                        else:
                            if len(input_list) == 1:
                                input_list_t = input_list[0]
                            else:
                                input_list_t = torch.cat(input_list, dim=0)
                            input_list_t2 = torch.cat([input_list_t, input_list_t], dim=0).to(torch.float16)
                            encoder_hidden_states2 = torch.cat([pos_caption_enc, neg_caption_enc], dim=0).to(torch.float16)
                            scale_params = {"s3diff_lora_scaling": 0.25}
                            if self.is_lyra_unet:
                                preds = self.unet(input_list_t2, encoder_hidden_states2, self.timesteps, map_extra_tensors, scale_params)
                            else:
                                preds = self.unet(input_list_t2, self.timesteps, encoder_hidden_states=encoder_hidden_states2).sample
                            torch.cuda.synchronize()
                            model_out = preds[1] + self.guidance_scale * (preds[0] - preds[1])
                        input_list = []
                    noise_preds.append(model_out)

            print("duration unet about", time.perf_counter()-t0)
            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred
            print("duration: ", time.perf_counter()-t0)

        # timedot0("before vae decode", beg_time)
        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        # print("vae decode shape:", x_denoised.shape)
        # vae decode shape: torch.Size([1, 4, 288, 160])
        # vae decode shape: torch.Size([1, 4, 288, 160])
        # devices.torch_gc()
        print("vae decode input shape", x_denoised.shape)
        if self.is_lyra_vae:
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            if hasattr(self, "vae_lyra"):
                del self.vae_lyra
            # print(x_denoised / self.vae.config.scaling_factor)
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        timedot0("after vae decode", beg_time)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}
        sd["w"] = self.W

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }

        torch.save(sd, outf)

    def _set_latent_tile(self,
        latent_tiled_size = 96,
        latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap
    
    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))

