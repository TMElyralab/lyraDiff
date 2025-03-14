cmake_minimum_required(VERSION 3.8)

add_library(ZeroCrossAttnWeight STATIC ZeroCrossAttnWeight.cc)

set_property(TARGET ZeroCrossAttnWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZeroCrossAttnWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZeroCrossAttnWeight PUBLIC cuda_utils logger memory_utils)

add_library(ZeroCrossAttn STATIC ZeroCrossAttn.cc)

set_property(TARGET ZeroCrossAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZeroCrossAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZeroCrossAttn PUBLIC add_bias residual group_norm basic_transformer_kernels flash_attn2 ZeroCrossAttnWeight activation cuda_utils logger memory_utils)
