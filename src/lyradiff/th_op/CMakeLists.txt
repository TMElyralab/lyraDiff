add_definitions(-DTORCH_CUDA=1)

add_library(th_utils STATIC th_utils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(th_utils PUBLIC "${TORCH_LIBRARIES}" -lcublas -lcudart -lcurand tensor)

add_subdirectory(unet_2d_conditional)
add_subdirectory(xlunet_2d_conditional)
add_subdirectory(vae_model)
add_subdirectory(glvunet_2d_conditional)
add_subdirectory(flux_transformer_2d_model)
add_subdirectory(lyradiff_common_context)
add_subdirectory(flux_controlnet_model)
add_subdirectory(flux_transformer_block)
add_subdirectory(flux_single_transformer_block)

add_library(pybind STATIC pybind.cc)
set_property(TARGET pybind PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET pybind PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(pybind PUBLIC "${TORCH_LIBRARIES}" -lcublas -lcudart -lcurand tensor th_flux_controlnet th_flux_single_transformer_block th_flux_transformer_block th_lyradiff_common_context th_flux_transformer)


add_library(th_lyradiff SHARED
            $<TARGET_OBJECTS:th_unet>
            $<TARGET_OBJECTS:th_vae>
            $<TARGET_OBJECTS:th_xlunet>
            $<TARGET_OBJECTS:th_glvunet>
            $<TARGET_OBJECTS:th_flux_transformer>
            $<TARGET_OBJECTS:th_lyradiff_common_context>
            $<TARGET_OBJECTS:th_flux_controlnet>
            $<TARGET_OBJECTS:th_flux_transformer_block>
            $<TARGET_OBJECTS:th_flux_single_transformer_block>
            $<TARGET_OBJECTS:pybind>
)
add_library(th_lyradiffxl SHARED
             $<TARGET_OBJECTS:th_xlunet>
)

target_link_libraries(th_lyradiff PUBLIC "${TORCH_LIBRARIES}"
th_unet
th_vae
th_xlunet
th_glvunet
th_flux_transformer
th_lyradiff_common_context
th_flux_transformer_block
th_flux_single_transformer_block
th_flux_controlnet
pybind
)

target_link_libraries(th_lyradiffxl PUBLIC "${TORCH_LIBRARIES}"
th_xlunet
)