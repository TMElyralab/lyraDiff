cmake_minimum_required(VERSION 3.8)

file(GLOB LAUNCH_CUDA_SOURCES *.cu)

add_library(flash_attn2 STATIC ${LAUNCH_CUDA_SOURCES})
set_property(TARGET flash_attn2 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flash_attn2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(flash_attn2 PUBLIC -lcudart)
