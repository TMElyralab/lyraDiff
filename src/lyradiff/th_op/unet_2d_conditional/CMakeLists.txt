add_library(th_unet STATIC Unet2dConditionalModelOp.cc)
set_property(TARGET th_unet PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_unet PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto memory_utils Unet2dConditionalModel cublasMMWrapper tensor cuda_utils th_utils)
