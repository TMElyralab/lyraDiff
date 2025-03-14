add_library(th_lyradiff_common_context STATIC LyraDiffCommonContext.cc)
set_property(TARGET th_lyradiff_common_context PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_lyradiff_common_context PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto common_ops memory_utils cublasMMWrapper tensor cuda_utils th_utils)
