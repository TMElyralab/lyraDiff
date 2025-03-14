cmake_minimum_required(VERSION 3.8)

add_library(ImageProjectBlockWeight STATIC ImageProjectWeight.cc)

set_property(TARGET ImageProjectBlockWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ImageProjectBlockWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ImageProjectBlockWeight PUBLIC -lcublas activation cuda_utils logger memory_utils)

add_library(ImageProjectBlock STATIC ImageProjectBlock.cc)

set_property(TARGET ImageProjectBlock PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ImageProjectBlock PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ImageProjectBlock PUBLIC -lcublas activation timestep_ffn_kernels cuda_utils logger gemm memory_utils ImageProjectBlockWeight)

