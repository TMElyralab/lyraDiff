import torch
import os

lib_path = os.getenv("LYRA_LIB_PATH", "/workspace/lyradiff_libs/libth_lyradiff.so") 
torch.classes.load_library(lib_path)
