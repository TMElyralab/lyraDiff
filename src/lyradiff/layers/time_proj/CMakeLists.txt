cmake_minimum_required(VERSION 3.8)

add_library(TimeProjection STATIC TimeProjection.cc)

set_property(TARGET TimeProjection PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TimeProjection PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TimeProjection PUBLIC -lcublas time_proj_kernels cuda_utils logger memory_utils)