cmake_minimum_required(VERSION 3.8)

add_library(ZeroSFTWeight STATIC ZeroSFTWeight.cc)

set_property(TARGET ZeroSFTWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZeroSFTWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZeroSFTWeight PUBLIC cuda_utils logger memory_utils)

add_library(ZeroSFT STATIC ZeroSFT.cu)

set_property(TARGET ZeroSFT PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZeroSFT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZeroSFT PUBLIC activation residual cat_kernels zero_sft_kernels group_norm conv2d ZeroSFTWeight activation cuda_utils logger memory_utils)