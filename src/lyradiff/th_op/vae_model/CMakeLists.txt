add_library(th_vae STATIC VaeModelOp.cc)
set_property(TARGET th_vae PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_vae PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto memory_utils VaeModel cublasMMWrapper tensor cuda_utils th_utils)
