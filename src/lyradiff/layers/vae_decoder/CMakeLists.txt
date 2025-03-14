cmake_minimum_required(VERSION 3.8)

add_library(VaeDecoderWeight STATIC VaeDecoderWeight.cc)
set_property(TARGET VaeDecoderWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeDecoderWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeDecoderWeight PUBLIC UpDecoderBlock2dWeight UNetMidBlock2DWeight)

add_library(VaeDecoder STATIC VaeDecoder.cc)
set_property(TARGET VaeDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeDecoder PUBLIC conv2d group_norm Resnet2DBlock UpDecoderBlock2d UNetMidBlock2D VaeDecoderWeight)
