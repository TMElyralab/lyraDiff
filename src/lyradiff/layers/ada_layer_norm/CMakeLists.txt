cmake_minimum_required(VERSION 3.8)

add_library(AdaLayerNormWeight STATIC AdaLayerNormWeight.cc)

set_property(TARGET AdaLayerNormWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AdaLayerNormWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AdaLayerNormWeight PUBLIC cuda_utils logger memory_utils load_lora)

add_library(AdaLayerNorm STATIC AdaLayerNorm.cc)

set_property(TARGET AdaLayerNorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AdaLayerNorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AdaLayerNorm PUBLIC AdaLayerNormWeight activation cuda_utils logger memory_utils layernorm common_ops)

add_library(AdaFP8LayerNormWeight STATIC AdaFP8LayerNormWeight.cc)

set_property(TARGET AdaFP8LayerNormWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AdaFP8LayerNormWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AdaFP8LayerNormWeight PUBLIC cuda_utils AdaLayerNormWeight logger memory_utils load_lora)

add_library(AdaFP8LayerNorm STATIC AdaFP8LayerNorm.cc)

set_property(TARGET AdaFP8LayerNorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AdaFP8LayerNorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AdaFP8LayerNorm PUBLIC AdaLayerNorm AdaFP8LayerNormWeight activation cuda_utils logger memory_utils layernorm common_ops)
