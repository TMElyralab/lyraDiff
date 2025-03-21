cmake_minimum_required(VERSION 3.8)

add_library(FluxTransformerBlockWeight STATIC FluxTransformerBlockWeight.cc)

set_property(TARGET FluxTransformerBlockWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerBlockWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerBlockWeight PUBLIC memory_utils FluxAttentionProcessorWeight FluxAttnPostProcessorWeight AdaLayerNormWeight cuda_utils load_lora logger memory_utils)

add_library(FluxTransformerBlock STATIC FluxTransformerBlock.cc)

set_property(TARGET FluxTransformerBlock PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerBlock PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerBlock PUBLIC FluxTransformerBlockWeight FluxAttentionProcessor FluxAttnPostProcessor AdaLayerNorm flux_single_transformer_kernels activation cuda_utils logger memory_utils flux_apply_rope common_ops)

add_library(FluxTransformerFP8BlockWeight STATIC FluxTransformerFP8BlockWeight.cc)

set_property(TARGET FluxTransformerFP8BlockWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerFP8BlockWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerFP8BlockWeight PUBLIC memory_utils FluxAttentionFP8ProcessorWeight FluxAttnPostFP8ProcessorWeight AdaFP8LayerNormWeight cuda_utils load_lora logger memory_utils)

add_library(FluxTransformerFP8Block STATIC FluxTransformerFP8Block.cc)

set_property(TARGET FluxTransformerFP8Block PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerFP8Block PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerFP8Block PUBLIC FluxTransformerFP8BlockWeight FluxTransformerBlock FluxAttentionFP8Processor FluxAttnPostFP8Processor AdaFP8LayerNorm flux_single_transformer_kernels activation cuda_utils logger memory_utils flux_apply_rope common_ops)


add_library(FluxTransformerInt4BlockWeight STATIC FluxTransformerInt4BlockWeight.cc)

set_property(TARGET FluxTransformerInt4BlockWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerInt4BlockWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerInt4BlockWeight PUBLIC memory_utils FluxAttentionInt4ProcessorWeight FluxAttnPostInt4ProcessorWeight AdaFP8LayerNormWeight cuda_utils load_lora logger memory_utils)

add_library(FluxTransformerInt4Block STATIC FluxTransformerInt4Block.cc)

set_property(TARGET FluxTransformerInt4Block PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET FluxTransformerInt4Block PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(FluxTransformerInt4Block PUBLIC FluxTransformerInt4BlockWeight FluxTransformerBlock FluxAttentionInt4Processor FluxAttnPostInt4Processor AdaFP8LayerNorm flux_single_transformer_kernels activation cuda_utils logger memory_utils flux_apply_rope common_ops)
