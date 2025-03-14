#pragma once
#include "src/lyradiff/utils/cuda_utils.h"
#include <src/lyradiff/interface/IFLoraWeight.h>
#include <string>
#include <vector>
namespace lyradiff {
template<typename T>
class IFBaseWeight {

protected:
    LyraQuantType quant_level_;
    bool          is_quantized_level_;

public:
    std::string weight_name_;
    // public:
    // std::vector<IFCBasicTransformerContainerWeight<T>*> vec_basic_transformer_container_weights;

    // virtual void
    // loadIPAdapterFromWeight(const std::string& ip_adapter_path, const float& scale, FtCudaDataType model_file_type)
    // {
    //     for (auto container_weight : vec_basic_transformer_container_weights)
    //         container_weight->loadIPAdapterFromWeight(ip_adapter_path, scale, model_file_type);
    // }

    // virtual void unLoadIPAdapter()
    // {
    //     for (auto container_weight : vec_basic_transformer_container_weights)
    //         container_weight->unLoadIPAdapter();
    // }
};
}  // namespace lyradiff