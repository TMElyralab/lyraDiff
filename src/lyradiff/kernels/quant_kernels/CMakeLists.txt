add_library(quant_kernels STATIC quant_kernels.cu)
set_property(TARGET quant_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET quant_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(SOURCE quant_kernels.cu APPEND PROPERTY INCLUDE_DIRECTORIES)

add_library(layer_norm_int8 STATIC layer_norm_int8.cu)
set_property(TARGET layer_norm_int8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET layer_norm_int8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(SOURCE layer_norm_int8.cu APPEND PROPERTY INCLUDE_DIRECTORIES)

add_library(ffn_kernels_int8 STATIC ffn_kernels_int8.cu)
set_property(TARGET ffn_kernels_int8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ffn_kernels_int8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(SOURCE ffn_kernels_int8.cu APPEND PROPERTY INCLUDE_DIRECTORIES)