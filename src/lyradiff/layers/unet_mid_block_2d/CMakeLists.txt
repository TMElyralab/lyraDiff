cmake_minimum_required(VERSION 3.8)

add_library(UNetMidBlock2DWeight STATIC UNetMidBlock2DWeight.cc)

set_property(TARGET UNetMidBlock2DWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UNetMidBlock2DWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UNetMidBlock2DWeight PUBLIC -lcublas Resnet2DBlockWeight cuda_utils logger memory_utils)

add_library(UNetMidBlock2D STATIC UNetMidBlock2D.cc)

set_property(TARGET UNetMidBlock2D PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UNetMidBlock2D PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UNetMidBlock2D PUBLIC -lcublas Resnet2DBlock basic_transformer_kernels add_bias softmax cuda_utils logger gemm memory_utils)