cmake_minimum_required(VERSION 3.8)

add_library(DownBlock2DWeight STATIC DownBlock2DWeight.cc)

set_property(TARGET DownBlock2DWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DownBlock2DWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DownBlock2DWeight PUBLIC -lcublas Resnet2DBlockWeight cuda_utils logger memory_utils)

add_library(DownBlock2D STATIC DownBlock2D.cc)

set_property(TARGET DownBlock2D PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DownBlock2D PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DownBlock2D PUBLIC -lcublas Resnet2DBlock cuda_utils logger gemm memory_utils)


add_library(XLDownBlock2DWeight STATIC XLDownBlock2DWeight.cc)

set_property(TARGET XLDownBlock2DWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET XLDownBlock2DWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(XLDownBlock2DWeight PUBLIC -lcublas Resnet2DBlockWeight cuda_utils logger memory_utils)

add_library(XLDownBlock2D STATIC XLDownBlock2D.cc)

set_property(TARGET XLDownBlock2D PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET XLDownBlock2D PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(XLDownBlock2D PUBLIC -lcublas Resnet2DBlock cuda_utils logger gemm memory_utils)