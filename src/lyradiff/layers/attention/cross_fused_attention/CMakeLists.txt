file(GLOB_RECURSE FMHCA_FILES *.cpp)
add_library(cross_attention STATIC ${FMHCA_FILES})

target_link_libraries(cross_attention PUBLIC -lcublas -lcudart cudaDriverWrapper)

set_property(TARGET cross_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cross_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)