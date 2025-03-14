add_library(basic_transformer_kernels STATIC basic_transformer_kernels.cu)
set_property(TARGET basic_transformer_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET basic_transformer_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(ffn_kernels STATIC ffn_kernels.cu)
set_property(TARGET ffn_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ffn_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(fused_geglu_kernel STATIC fused_geglu_kernel.cu)
set_property(TARGET fused_geglu_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fused_geglu_kernel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(SOURCE fused_geglu_kernel.cu APPEND PROPERTY INCLUDE_DIRECTORIES ${CUTLASS_DUAL_GEMM_DIR})

add_library(int8SQ STATIC fused_geglu_kernel.cu)
set_property(TARGET int8SQ PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET int8SQ PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(SOURCE int8SQ.cu APPEND PROPERTY INCLUDE_DIRECTORIES ${CUTLASS_DUAL_GEMM_DIR})