cmake_minimum_required(VERSION 3.8)

add_library(conv2d STATIC conv2d.cc)
set_property(TARGET conv2d PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET conv2d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(conv2d PUBLIC -lcublas -lcudart -lcudnn -lcublasLt logger tensor memory_utils residual cublasMMWrapper) #cublasMMWrapper)