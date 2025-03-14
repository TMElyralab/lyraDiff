
file(GLOB_RECURSE FMHA_FILES *.cpp)
add_library(flash_attention_v2 STATIC ${FMHA_FILES})

target_link_libraries(flash_attention_v2 PUBLIC -lcublas -lcudart cuda_utils logger cublasAlgoMap cublasMMWrapper cudaDriverWrapper tensor  memory_utils lyra_exception)

set_property(TARGET flash_attention_v2 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flash_attention_v2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)