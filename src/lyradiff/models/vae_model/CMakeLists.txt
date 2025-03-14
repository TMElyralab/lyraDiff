cmake_minimum_required(VERSION 3.8)


add_library(VaeModelWeight STATIC VaeModelWeight.cc)

set_property(TARGET VaeModelWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeModelWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeModelWeight PUBLIC  cuda_utils memory_utils VaeDecoderWeight)

add_library(VaeModel STATIC VaeModel.cc)

set_property(TARGET VaeModel PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET VaeModel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(VaeModel PUBLIC conv2d VaeModelWeight VaeDecoder VaeEncoder cuda_utils logger memory_utils)
