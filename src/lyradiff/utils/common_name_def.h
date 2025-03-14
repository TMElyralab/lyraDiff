#pragma once 

#include <iostream>
#include <string>
#include <vector>
namespace lyradiff {
// tensor name
const std::string HIDDEN_STATES         = "hidden_states";
const std::string ENCODER_HIDDEN_STATES = "encoder_hidden_states";
const std::string IP_HIDDEN_STATES      = "ip_hidden_states";
const std::string IP_SHALLOW_MASK       = "ip_shallow_mask";
const std::string IP_DEEP_MASK          = "ip_deep_mask";
const std::string PROMPT_IMAGE_EMB      = "prompt_image_emb";

const std::string TEMB             = "temp";
const std::string CONDITIONING_IMG = "conditioning_img";

const std::string NM_INPUT  = "input";
const std::string NM_OUTPUT = "output";

const std::vector<std::string> COMMON_INP_NAMES = {IP_HIDDEN_STATES};

// param name
const std::string IP_RATIO     = "ip_ratio";
}  // namespace lyradiff