cmake_minimum_required(VERSION 3.8 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
project(LyraDiff LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)

# enable BF16
if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")
endif()

if((${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11" AND ${CUDA_VERSION_MINOR} VERSION_GREATER_EQUAL "8") OR (${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "12"))
  add_definitions("-DENABLE_FP8")
  option(ENABLE_FP8 "ENABLE_FP8" OFF)
  if(ENABLE_FP8)
    message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
  endif()
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL "10.1")
  add_definitions(-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(CUDNN)
if(NOT CUDNN_FOUND)
  message(FATAL_ERROR "Didn't found package CUDNN!")
endif()

option(BUILD_CUTLASS_MOE "Builds CUTLASS kernels supporting MoE GEMM" ON)
if(BUILD_CUTLASS_MOE)
  message(STATUS "Add DBUILD_CUTLASS_MOE, requires CUTLASS. Increases compilation time")
  add_definitions("-DBUILD_CUTLASS_MOE")
endif()

option(BUILD_CUTLASS_MIXED_GEMM "Builds CUTLASS kernels supporting mixed gemm" ON)
if(BUILD_CUTLASS_MIXED_GEMM)
  message(STATUS "Add DBUILD_CUTLASS_MIXED_GEMM, requires CUTLASS. Increases compilation time")
  add_definitions("-DBUILD_CUTLASS_MIXED_GEMM")
endif()

option(BUILD_PYT "Build in PyTorch TorchScript class mode" OFF)
# option(THRUST_IGNORE_CUB_VERSION_CHECK ON)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  if(GIT_AUTOCLONE_CUTLASS)
    message(STATUS "Running submodule update to fetch cutlass")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/cutlass
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init 3rdparty/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
    endif()
  endif()
endif()

set(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)
set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include)
#set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass)
set(CUTLASS_UTIL_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include)
set(CUTLASS_DUAL_GEMM_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/examples/45_dual_gemm)
set(CUTLASS_CONTRIB_DIR ${PROJECT_SOURCE_DIR}/cutlass_contrib/include)
set(CUTLASS_EXTENSIONS_DIR ${PROJECT_SOURCE_DIR}/src/lyradiff/kernels/cutlass_extensions/include)


option(SPARSITY_SUPPORT "Build project with Ampere sparsity feature support" OFF)

option(BUILD_FAST_MATH "Build in fast math mode" ON)

if(BUILD_MULTI_GPU)
  message(STATUS "Add DBUILD_MULTI_GPU, requires MPI and NCCL")
  add_definitions("-DBUILD_MULTI_GPU")
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
  set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building
endif()

if(BUILD_PYT)
  if(DEFINED ENV{NVIDIA_PYTORCH_VERSION})
    if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_LESS "20.03")
      message(FATAL_ERROR "NVIDIA PyTorch image is too old for TorchScript mode.")
    endif()
    if($ENV{NVIDIA_PYTORCH_VERSION} VERSION_EQUAL "20.03")
      add_definitions(-DLEGACY_THS=1)
    endif()
  endif()
endif()

if(USE_TRITONSERVER_DATATYPE)
  message("-- USE_TRITONSERVER_DATATYPE")
  add_definitions("-DUSE_TRITONSERVER_DATATYPE")
endif()

set(CXX_STD "17" CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
message(${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# profiling
option(USE_NVTX "Whether or not to use nvtx" ON)
if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl") # -Xptxas -v

set(SM_SETS 52 60 61 70 75 80 86 89 90)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86 OR SM_NUM STREQUAL 89 OR SM_NUM STREQUAL 90)
      set(USING_WMMA True)
    endif()

    if(BUILD_PYT)
      string(SUBSTRING ${SM_NUM} 0 1 SM_MAJOR)
      string(SUBSTRING ${SM_NUM} 1 1 SM_MINOR)
      set(ENV{TORCH_CUDA_ARCH_LIST} "$ENV{TORCH_CUDA_ARCH_LIST}\;${SM_MAJOR}.${SM_MINOR}")
    endif()

    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  #                      -rdc=true")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0;8.6")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()

if(BUILD_PYT)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wunused-but-set-variable")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
if(BUILD_FAST_MATH)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
message("CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# # set NVCC
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --use_fast_math -rdc=true -lcudadevrt")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
  ${CUTLASS_HEADER_DIR}
  ${CUTLASS_UTIL_HEADER_DIR}
  ${CUTLASS_CONTRIB_DIR}
  ${CUTLASS_DUAL_GEMM_DIR}
  ${CUTLASS_EXTENSIONS_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty  # add cub include files
)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if (SPARSITY_SUPPORT)
  list(APPEND COMMON_HEADER_DIRS ${CUSPARSELT_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${CUSPARSELT_PATH}/lib64)
  add_definitions(-DSPARSITY_ENABLED=1)
endif()

if (BUILD_MULTI_GPU)
  list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})
  list(APPEND COMMON_LIB_DIRS /usr/local/mpi/lib)
endif()

if(USE_TRITONSERVER_DATATYPE)
  list(APPEND COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/../repo-core-src/include)
endif()

set(PYTHON_PATH "python" CACHE STRING "Python path")
if(BUILD_PYT)
  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_VERSION)
  if (TORCH_VERSION VERSION_LESS "1.5.0")
      message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
  endif()
  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  find_package(Torch REQUIRED)
  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE PY_INCLUDE_DIR)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Python config Error.")
  endif()
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE USE_CXX11_ABI)
  message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
  if (USE_CXX11_ABI)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
endif()

include_directories(
  ${COMMON_HEADER_DIRS},
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(3rdparty)
add_subdirectory(src)
# add_subdirectory(examples)

# add_subdirectory(tests)

# # Mesaure the compile time
option(MEASURE_BUILD_TIME "Measure the build time of each module" OFF)
if (MEASURE_BUILD_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif()

########################################

add_library(transformer-shared SHARED
  $<TARGET_OBJECTS:cublasAlgoMap>
  $<TARGET_OBJECTS:cublasMMWrapper>
  $<TARGET_OBJECTS:cuda_utils>
  $<TARGET_OBJECTS:logger>
  $<TARGET_OBJECTS:tensor>
)

if (BUILD_MULTI_GPU)
target_link_libraries(transformer-shared PUBLIC
  -lmpi
  ${NCCL_LIBRARIES}
)
endif()

if(USE_NVTX)
target_link_libraries(transformer-shared PUBLIC
  -lnvToolsExt
)
endif()
  
set_target_properties(transformer-shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(transformer-shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(transformer-shared PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(transformer-shared PUBLIC -lcudart -lcublas -lcublasLt -lcurand)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/LyraDiff)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/LyraDiffConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LyraDiffConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/LyraDiffConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  TARGETS
    transformer-shared
  EXPORT
    transformer-shared-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/lyradiff
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/lyradiff
)

install(
  EXPORT
    transformer-shared-targets
  FILE
    LyraDiffTargets.cmake
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

export(
  EXPORT
    transformer-shared-targets
  FILE
    ${CMAKE_CURRENT_BINARY_DIR}/LyraDiffTargets.cmake
  NAMESPACE
    TritonCore::
)

export(PACKAGE LyraDiff)
