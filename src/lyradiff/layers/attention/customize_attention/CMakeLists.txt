set(lib_files
  attention_layer.cu
  attention_fused_short.cu
  attention_fused_long.cu
  attention_nofused_utils.cu
  attention_nofused.cu
  cutlass_attention.cu
)

add_library(customize_attention STATIC ${lib_files})

target_link_libraries(customize_attention PUBLIC -lcublas -lcudart gemm)

set_property(TARGET customize_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET customize_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)