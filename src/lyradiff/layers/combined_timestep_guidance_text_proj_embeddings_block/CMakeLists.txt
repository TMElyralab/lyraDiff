add_library(CombinedTimestepGuidanceTextProjEmbeddingsWeight STATIC CombinedTimestepGuidanceTextProjEmbeddingsWeight.cc)

set_property(TARGET CombinedTimestepGuidanceTextProjEmbeddingsWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CombinedTimestepGuidanceTextProjEmbeddingsWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CombinedTimestepGuidanceTextProjEmbeddingsWeight PUBLIC -lcublas activation cuda_utils logger memory_utils load_lora TimestepEmbeddingBlockWeight)

add_library(CombinedTimestepGuidanceTextProjEmbeddings STATIC CombinedTimestepGuidanceTextProjEmbeddings.cc)

set_property(TARGET CombinedTimestepGuidanceTextProjEmbeddings PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CombinedTimestepGuidanceTextProjEmbeddings PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CombinedTimestepGuidanceTextProjEmbeddings PUBLIC -lcublas activation timestep_ffn_kernels TimestepEmbeddingBlock TimeProjection CombinedTimestepGuidanceTextProjEmbeddingsWeight cuda_utils logger gemm memory_utils common_ops)