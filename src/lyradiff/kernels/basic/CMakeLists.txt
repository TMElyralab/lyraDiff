add_library(gemm STATIC gemm.cu)
set_property(TARGET gemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(common_ops STATIC common_ops.cu)
set_property(TARGET common_ops PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET common_ops PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)