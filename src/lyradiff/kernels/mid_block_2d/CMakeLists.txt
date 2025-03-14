add_library(add_bias STATIC add_bias.cu)
set_property(TARGET add_bias PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET add_bias PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(softmax STATIC softmax.cu)
set_property(TARGET softmax PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET softmax PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)