cmake_minimum_required(VERSION 3.8)

add_library(W4A4GemmWeight STATIC W4A4GemmWeight.cc)
set_property(TARGET W4A4GemmWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET W4A4GemmWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(W4A4GemmWeight PRIVATE load_lora memory_utils cuda_utils logger cublasMMWrapper)

add_library(W4A4Gemm STATIC W4A4Gemm.cc)
set_property(TARGET W4A4Gemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET W4A4Gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(W4A4Gemm PRIVATE cublasMMWrapper W4A4GemmWeight w4a4_gemm memory_utils cuda_utils logger)
