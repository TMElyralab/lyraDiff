import torch
from enum import Enum

class LyraQuantLevel(Enum):
    NONE                     = 0
    FP8_W8A8                 = 1
    FP8_W8A8_FULL            = 2
    FP8_WEIGHT_ONLY          = 3
    INT8_SMOOTH_QUANT_LEVEL1 = 4
    INT8_SMOOTH_QUANT_LEVEL2 = 5
    INT8_SMOOTH_QUANT_LEVEL3 = 6
    INT4_W4A4                = 7
    INT4_W4A4_FULL           = 8

lyradiff_context = None 

def get_lyradiff_context():
    global lyradiff_context
    if lyradiff_context is not None:
        return lyradiff_context
    lyradiff_context = torch.classes.lyradiff.LyraDiffCommonContext()
    return lyradiff_context

def get_aug_emb(add_time_proj_model, add_embedding_model, time_ids, text_embeds, dtype):
    time_embeds = add_time_proj_model(time_ids.flatten())
    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
    add_embeds = add_embeds.to(dtype)
    aug_emb = add_embedding_model(add_embeds)
    return aug_emb


def load_embedding_weight(model, state_dict):
    sub_state_dict = {}
    for k in state_dict:
        if k.startswith("add_embedding"):
            v = state_dict[k]
            sub_k = ".".join(k.split(".")[1:])
            sub_state_dict[sub_k] = v

    model.load_state_dict(sub_state_dict)

