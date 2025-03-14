
file(GLOB_RECURSE FMHA_FILES *.cpp)
add_library(flash_attention_v1 STATIC ${FMHA_FILES})

target_link_libraries(flash_attention_v1 PUBLIC -lcublas -lcudart cudaDriverWrapper)

set_property(TARGET flash_attention_v1 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flash_attention_v1 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)