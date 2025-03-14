add_library(load_lora STATIC load_lora.cu)
set_property(TARGET load_lora PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET load_lora PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)