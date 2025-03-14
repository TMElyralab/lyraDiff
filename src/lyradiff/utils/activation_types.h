#pragma once

#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

enum class ActivationType {
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

inline ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return ActivationType::GeGLU;
    }
    else if (activation_type_str == "ReGLU" || activation_type_str == "reglu" || activation_type_str == "gated-relu") {
        return ActivationType::ReGLU;
    }
    else if (activation_type_str == "SiGLU" || activation_type_str == "gated-silu") {
        return ActivationType::SiGLU;
    }
    else {
        LYRA_CHECK_WITH_INFO(false, "Activation Type: " + activation_type_str + " not supported !");
    }
    return ActivationType::InvalidType;
}

inline bool isGatedActivation(ActivationType activaiton_type)
{
    return activaiton_type == ActivationType::GeGLU || activaiton_type == ActivationType::ReGLU
           || activaiton_type == ActivationType::SiGLU;
}

}  // namespace lyradiff