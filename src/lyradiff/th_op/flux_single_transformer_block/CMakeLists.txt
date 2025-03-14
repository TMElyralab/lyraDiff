add_library(th_flux_single_transformer_block STATIC FluxSingleTransformerBlockOp.cc)
set_property(TARGET th_flux_single_transformer_block PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_flux_single_transformer_block PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto th_lyradiff_common_context common_ops memory_utils FluxTransformer2DModel cublasMMWrapper tensor cuda_utils th_utils)
