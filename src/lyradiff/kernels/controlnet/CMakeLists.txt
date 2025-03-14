
add_library(residual STATIC residual.cu)
set_property(TARGET residual PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET residual PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)