
add_library(interpolate STATIC interpolate.cu)
set_property(TARGET interpolate PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET interpolate PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(interpolate PUBLIC cuda_utils)