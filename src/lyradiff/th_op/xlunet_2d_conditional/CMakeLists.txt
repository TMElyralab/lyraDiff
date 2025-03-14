add_library(th_xlunet STATIC XLUnet2dConditionalModelOp.cc)
set_property(TARGET th_xlunet PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_xlunet PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto common_ops memory_utils XLUnet2dConditionalModel cublasMMWrapper tensor cuda_utils th_utils)
