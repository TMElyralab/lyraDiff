add_library(group_norm STATIC group_norm.cu)
set_property(TARGET group_norm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET group_norm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(layernorm STATIC layer_norm.cu)
set_property(TARGET layernorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layernorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(rms_norm STATIC rms_norm.cu)
set_property(TARGET rms_norm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rms_norm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

# target_compile_options(layernorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        --generate-line-info
#                        --use_fast_math
#                        --relocatable-device-code=true
#                        >)
