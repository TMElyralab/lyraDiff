
add_library(time_proj_kernels STATIC time_proj_kernels.cu)
set_property(TARGET time_proj_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET time_proj_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)