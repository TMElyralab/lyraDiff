add_library(down_encoder_block_2d_kernels STATIC down_encoder_block_2d_kernels.cu)
set_property(TARGET down_encoder_block_2d_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET down_encoder_block_2d_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)