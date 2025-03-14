#pragma once
#include "src/lyradiff/utils/Tensor.h"

namespace lyradiff {
class LyraDiffContext: public TensorMap {
public:
    // TensorMap                              map_tensors;
    std::string                            cur_running_module;
    bool                                   is_controlnet = false;
    std::unordered_map<std::string, float> map_scale_params;

    float getParamVal(const std::string& name)
    {
        if (map_scale_params.find(name) != map_scale_params.end())
            return map_scale_params.at(name);
        return 0;
    }
};

};  // namespace lyradiff