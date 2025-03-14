
add_library(cat_kernels STATIC cat_kernels.cu)
set_property(TARGET cat_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cat_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)