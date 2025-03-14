add_library(Resnet2DBlockWeight STATIC Resnet2DBlockWeight.cc)
target_link_libraries(Resnet2DBlockWeight PUBLIC memory_utils load_lora)
set_property(TARGET Resnet2DBlockWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Resnet2DBlockWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(Resnet2DBlock STATIC Resnet2DBlock.cu)
target_link_libraries(Resnet2DBlock PUBLIC -lcublas -lcudart -lcudnn -lcublasLt activation group_norm resnet2d_kernels conv2d Resnet2DBlockWeight)
set_property(TARGET Resnet2DBlock PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Resnet2DBlock PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)