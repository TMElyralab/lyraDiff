#pragma once
#include "src/lyradiff/utils/cuda_utils.h"
#include <src/lyradiff/interface/IFLoraWeight.h>
#include <string>
#include <vector>
namespace lyradiff {
template<typename T>
class IFCBasicTransformerContainerWeight: public IFLoraWeight<T> {
    // class IFCBasicTransformerContainerWeight {
protected:
    std::vector<IFCBasicTransformerContainerWeight<T>*> vec_basic_transformer_container_weights;

public:
    // std::vector<IFCBasicTransformerContainerWeight<T>*> vec_basic_transformer_container_weights;

    virtual void
    loadIPAdapterFromWeight(const std::string& ip_adapter_path, const float& scale, FtCudaDataType model_file_type)
    {
        for (auto container_weight : vec_basic_transformer_container_weights)
            container_weight->loadIPAdapterFromWeight(ip_adapter_path, scale, model_file_type);
    }

    virtual void unLoadIPAdapter()
    {
        for (auto container_weight : vec_basic_transformer_container_weights)
            container_weight->unLoadIPAdapter();
    }
};
}  // namespace lyradiff