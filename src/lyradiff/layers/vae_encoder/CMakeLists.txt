cmake_minimum_required(VERSION 3.8)

add_library(VaeEncoderWeight STATIC VaeEncoderWeight.cc)
set_property(TARGET VaeEncoderWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeEncoderWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeEncoderWeight PUBLIC DownEncoderBlock2DWeight UNetMidBlock2DWeight)

add_library(VaeEncoder STATIC VaeEncoder.cc)
set_property(TARGET VaeEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeEncoder PUBLIC conv2d group_norm Resnet2DBlock DownEncoderBlock2D UNetMidBlock2D VaeEncoderWeight)
