add_library(flux_apply_rope STATIC flux_apply_rope.cu)
set_property(TARGET flux_apply_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flux_apply_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
