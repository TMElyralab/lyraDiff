add_library(th_glvunet STATIC GLVUnet2dConditionalModelOp.cc)
set_property(TARGET th_glvunet PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_glvunet PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto common_ops memory_utils GLVUnet2dConditionalModel cublasMMWrapper tensor cuda_utils th_utils)
