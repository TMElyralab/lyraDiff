add_library(th_flux_controlnet STATIC FluxControlnetModelOp.cc)
set_property(TARGET th_flux_controlnet PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_flux_controlnet PRIVATE "${TORCH_LIBRARIES}" -lcublas -lcublasLt -lcudnn -lnvidia-ml -lssl -lcrypto th_lyradiff_common_context common_ops memory_utils FluxControlnetModel cublasMMWrapper tensor cuda_utils th_utils)
