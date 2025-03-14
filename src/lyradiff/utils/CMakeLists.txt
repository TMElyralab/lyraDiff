cmake_minimum_required(VERSION 3.8)

add_library(cuda_utils STATIC cuda_utils.cc)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_utils PUBLIC -lcudart)

add_library(logger STATIC logger.cc)
set_property(TARGET logger PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logger PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(logger PUBLIC -lcudart)

add_library(cublasAlgoMap STATIC cublasAlgoMap.cc)
set_property(TARGET cublasAlgoMap PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasAlgoMap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasAlgoMap PUBLIC -lcublas -lcudart -lcurand cuda_utils logger)

add_library(cublasMMWrapper STATIC cublasMMWrapper.cc)
set_property(TARGET cublasMMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasMMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasMMWrapper PUBLIC -lcublas -lcudart -lcublasLt -lcurand smooth_quant_matmul cublasAlgoMap cuda_utils logger residual)
if (SPARSITY_SUPPORT)
target_link_libraries(cublasMMWrapper PUBLIC memory_utils -lcusparse -lcusparseLt)
endif()

add_library(cudaDriverWrapper STATIC cudaDriverWrapper.cc)
set_property(TARGET cudaDriverWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cudaDriverWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cudaDriverWrapper PUBLIC -lcublas -lcudart -lcurand cublasAlgoMap cuda_utils logger)

add_library(memory_utils STATIC memory_utils.cu)
set_property(TARGET memory_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memory_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(memory_utils PUBLIC cuda_utils logger)

add_library(lyra_exception STATIC lyraException.cc)
set_property(TARGET lyra_exception PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET lyra_exception PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(tensor STATIC Tensor.cc tensor_helper.cu)
set_property(TARGET tensor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET tensor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(tensor PUBLIC cuda_utils logger lyra_exception memory_utils)


add_library(test_utils STATIC test_utils.cc)
set_property(TARGET test_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET test_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(test_utils PUBLIC cuda_utils logger lyra_exception)