from safetensors.torch import load_file
import torch
from peft import LoraConfig, get_peft_model
import sys, os
import numpy as np

lib_path = os.getenv("LYRA_LIB_PATH", "/workspace/lyradiff_libs/libth_lyradiff.so") 
torch.classes.load_library(lib_path)

def is_attn1_q(name):
    return ("attn1" in name and "to_q" in name) or ("attentions.0.to_q" in name)
    

def is_attn2_k(name):
    return "attn2" in name and "to_k" in name

def get_qkv_weight(state_dict, k):
    if is_attn1_q(k):
        name_q = k
        name_k = k.replace("to_q", "to_k")
        name_v = k.replace("to_q", "to_v")
        qq,kk,vv = state_dict[name_q].to(torch.float16), state_dict[name_k].to(torch.float16), state_dict[name_v].to(torch.float16)
        return qq,kk,vv
    elif is_attn2_k(k):
        name_k = k
        name_v = k.replace("to_k", "to_v")
        kk,vv = state_dict[name_k].to(torch.float16), state_dict[name_v].to(torch.float16)
        return kk, vv
        

def get_processed_tensor(state_dict, k, v, is_alpha):
    if is_attn1_q(k):
        qq,kk,vv = get_qkv_weight(state_dict, k)
        if is_alpha:
            return torch.cat([qq,kk,vv],dim=0).cuda().to(torch.float16).contiguous()
        else:
            zero = torch.zeros_like(kk, dtype=kk.dtype).to(kk.device)
            res = torch.cat(
                [
                    torch.cat([qq, zero, zero], dim=1),
                    torch.cat([zero, kk, zero], dim=1),
                    torch.cat([zero, zero, vv], dim=1)
                ], dim = 0
            ).cuda().to(torch.float16).contiguous()
            return res
    elif is_attn2_k(k):
        kk,vv = get_qkv_weight(state_dict, k)
        if is_alpha:
            res = torch.cat([kk.unsqueeze(0),vv.unsqueeze(0)], dim=0).cuda().to(torch.float16).contiguous()
        else:
            zero = torch.zeros_like(kk, dtype=kk.dtype).to(kk.device)
            res = torch.cat(
                [
                    torch.cat([kk, zero], dim=1),
                    torch.cat([zero, vv], dim=1)
                ], dim = 0
            ).cuda().to(torch.float16).contiguous()
            return res
        v_concat = torch.cat([kk.unsqueeze(0),vv.unsqueeze(0)], dim=0).to(torch.float16).contiguous()
        return v_concat
    else:
        if len(v.shape) == 4:
            v = v.permute([0,2,3,1]).contiguous()
        v = v.cuda().to(torch.float16).contiguous()
        return v

def load_s3diff_lora(state_dict, rank):
    lora_alpha = 8
    weight_scale = lora_alpha / rank

    dict_lora_A = {}
    dict_lora_B = {}
    dict_others = {}
    for k in state_dict:
        v = state_dict[k]
            
        if "lora_A" in k:
            name = k.split(".lora_A")[0] + ".weight"
            vv = get_processed_tensor(state_dict, k, v, True)
            dict_lora_A[name] = vv
        elif "lora_B" in k:
            name = k.split(".lora_B")[0] + ".weight"
            vv = get_processed_tensor(state_dict, k, v, False)
            dict_lora_B[name] = vv * weight_scale
        else:
            dict_others[k] = v
    for k in dict_others:
        state_dict[k] = dict_others[k]
    
    return dict_lora_A, dict_lora_B, weight_scale

def get_de_mod_qkv(de_mod):
    zero = torch.zeros_like(de_mod, dtype=de_mod.dtype, device=de_mod.device)
    res = torch.cat(
        [
            torch.cat([de_mod, zero, zero], dim=1),
            torch.cat([zero, de_mod, zero], dim=1),
            torch.cat([zero, zero, de_mod], dim=1),
        ], dim=0
    ).to(torch.float16).contiguous()
    return res

def get_de_mod_kv(de_mod):
    zero = torch.zeros_like(de_mod, dtype=de_mod.dtype, device=de_mod.device)
    return torch.cat(
        [
            torch.cat([de_mod, zero], dim=1),
            torch.cat([zero, de_mod], dim=1)
        ], dim=0
    ).to(torch.float16).contiguous()

def parse_unet_embeds(unet_embeds, is_vae=False):
    unet_embeds = unet_embeds.to("cuda").to(torch.float16)
    rank = int(unet_embeds.shape[-1] ** 0.5)
    d_unet_de_mod = {}
    for i in range(4):
        prefix = "down_blocks." + str(i)
        emb = unet_embeds[:, i]
        prefix = prefix if not is_vae else "encoder."+prefix
        d_unet_de_mod[prefix] = emb
    prefix = "mid_block" if not is_vae else "encoder.mid_block"
    d_unet_de_mod[prefix] = unet_embeds[:, 4]
    if not is_vae:
        for i in range(4):
            prefix = "up_blocks." + str(i)
            prefix = prefix if not is_vae else "encoder."+prefix
            emb = unet_embeds[:, i+5]
            d_unet_de_mod[prefix] = emb
    d_unet_de_mod["other"] = unet_embeds[:, -1]    
    d_unet_de_mod2 = {}
    for k in d_unet_de_mod:
        d_unet_de_mod2[k] = d_unet_de_mod[k].reshape([rank, rank]).permute([1,0]).to(torch.float16).contiguous()
        d_unet_de_mod2[k+"_attn1_qkv"] = get_de_mod_qkv(d_unet_de_mod2[k]).contiguous()
        d_unet_de_mod2[k+"_attn2_kv"] = get_de_mod_kv(d_unet_de_mod2[k]).contiguous()
    
    return d_unet_de_mod2



class LyraUnet:
    def __init__(self, unet, config):
        self.unet = unet
        self.config = config

    def __call__(self, inp, enc, timesteps, map_extra_tensors, scale_params):
        scale_params = {"s3diff_lora_scaling": 0.25}
        inp2 = inp.to(torch.float16).permute([0,2,3,1]).contiguous()
        enc = enc.to(torch.float16)
        preds = self.unet.unet_forward(inp2,  enc, timesteps, None, map_extra_tensors, scale_params)
        res = preds.permute([0,3,1,2]).contiguous()
        return res