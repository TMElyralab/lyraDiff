cmake_minimum_required(VERSION 3.8)

add_library(ControlNetConditioningEmbeddingWeight STATIC ControlNetConditioningEmbeddingWeight.cc)

set_property(TARGET ControlNetConditioningEmbeddingWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ControlNetConditioningEmbeddingWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ControlNetConditioningEmbeddingWeight PUBLIC cuda_utils logger memory_utils)

add_library(ControlNetConditioningEmbedding STATIC ControlNetConditioningEmbedding.cu)

set_property(TARGET ControlNetConditioningEmbedding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ControlNetConditioningEmbedding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ControlNetConditioningEmbedding PUBLIC -lcublas conv2d ControlNetConditioningEmbeddingWeight activation cuda_utils logger memory_utils)