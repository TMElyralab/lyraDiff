cmake_minimum_required(VERSION 3.8)


add_library(FluxTransformer2DModelWeight STATIC FluxTransformer2DModelWeight.cc)

set_property(TARGET FluxTransformer2DModelWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformer2DModelWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformer2DModelWeight PUBLIC cuda_utils memory_utils FluxSingleTransformerFP8BlockWeight FluxTransformerFP8BlockWeight FluxSingleTransformerBlockWeight FluxTransformerBlockWeight AdaLayerNormWeight CombinedTimestepGuidanceTextProjEmbeddingsWeight)

add_library(FluxTransformer2DModel STATIC FluxTransformer2DModel.cc)

set_property(TARGET FluxTransformer2DModel PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformer2DModel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformer2DModel PUBLIC conv2d residual FluxTransformer2DModelWeight FluxSingleTransformerFP8Block FluxSingleTransformerInt4Block FluxTransformerFP8Block FluxTransformerInt4Block AdaLayerNorm CombinedTimestepGuidanceTextProjEmbeddings FluxTransformerBlock FluxSingleTransformerBlock cuda_utils logger memory_utils)
