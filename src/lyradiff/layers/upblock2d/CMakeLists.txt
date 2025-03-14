cmake_minimum_required(VERSION 3.8)

add_library(UpBlock2dWeight STATIC UpBlock2dWeight.cc)
set_property(TARGET UpBlock2dWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UpBlock2dWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UpBlock2dWeight PUBLIC Resnet2DBlockWeight load_lora)

add_library(UpBlock2d STATIC UpBlock2d.cc)
set_property(TARGET UpBlock2d PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET UpBlock2d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(UpBlock2d PUBLIC conv2d interpolate cat_kernels Resnet2DBlock UpBlock2dWeight)

add_library(XLUpBlock2dWeight STATIC XLUpBlock2dWeight.cc)
set_property(TARGET XLUpBlock2dWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET XLUpBlock2dWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(XLUpBlock2dWeight PUBLIC Resnet2DBlockWeight)

add_library(XLUpBlock2d STATIC XLUpBlock2d.cc)
set_property(TARGET XLUpBlock2d PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET XLUpBlock2d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(XLUpBlock2d PUBLIC conv2d interpolate cat_kernels Resnet2DBlock XLUpBlock2dWeight)

add_library(GLVUpBlock2dWeight STATIC GLVUpBlock2dWeight.cc)
set_property(TARGET GLVUpBlock2dWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GLVUpBlock2dWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GLVUpBlock2dWeight PUBLIC ZeroSFTWeight ZeroCrossAttnWeight Resnet2DBlockWeight)

add_library(GLVUpBlock2d STATIC GLVUpBlock2d.cc)
set_property(TARGET GLVUpBlock2d PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GLVUpBlock2d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GLVUpBlock2d PUBLIC ZeroSFT conv2d interpolate cat_kernels Resnet2DBlock GLVUpBlock2dWeight)
