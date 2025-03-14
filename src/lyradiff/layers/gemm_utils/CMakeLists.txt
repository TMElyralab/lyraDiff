add_definitions(-DTORCH_CUDA=1)

add_library(gemm_profiler STATIC GemmProfiler.cc)
set_property(TARGET gemm_profiler PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gemm_profiler PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(gemm_profiler PRIVATE cublasMMWrapper cutlass2_src cutlass3_src memory_utils cuda_utils logger memory_utils th_utils)

add_library(cutlass_matmul STATIC GemmProfiler.cc Matmul.cc)
set_property(TARGET cutlass_matmul PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cutlass_matmul PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cutlass_matmul PRIVATE cublasMMWrapper cutlass2_src cutlass3_src memory_utils cuda_utils logger memory_utils th_utils)

add_library(weightonly_matmul STATIC WeightOnlyMatmul.cc)
set_property(TARGET weightonly_matmul PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET weightonly_matmul PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(weightonly_matmul PRIVATE cublasMMWrapper gemm_profiler cutlass2_src cutlass3_src memory_utils cuda_utils logger memory_utils th_utils)

add_library(smooth_quant_matmul STATIC SmoothQuantGemm.cc)
set_property(TARGET smooth_quant_matmul PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET smooth_quant_matmul PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(smooth_quant_matmul PRIVATE cublasMMWrapper gemm_profiler int8SQ cutlass2_src cutlass3_src memory_utils cuda_utils logger memory_utils th_utils)
