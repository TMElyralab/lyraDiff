add_library(attn_cudriver_wrapper STATIC cudaDriverWrapper.cpp)

target_link_libraries(attn_cudriver_wrapper PUBLIC -lcublas -lcudart)

set_property(TARGET attn_cudriver_wrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET attn_cudriver_wrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)