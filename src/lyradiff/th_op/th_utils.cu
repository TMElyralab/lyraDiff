#include "src/lyradiff/th_op/th_utils.h"

namespace torch_ext {

std::vector<size_t> convert_shape(torch::Tensor tensor)
{
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.dim(); i++) {
        v_shape.push_back(tensor.size(i));
    }
    return v_shape;
}

template<typename T>
lyradiff::Tensor convert_tensor(torch::Tensor tensor)
{
    lyradiff::MemoryType mtype = tensor.is_cuda() ? lyradiff::MEMORY_GPU : lyradiff::MEMORY_CPU;
    return convert_tensor<T>(tensor, mtype);
}

template lyradiff::Tensor convert_tensor<int8_t>(torch::Tensor tensor);
template lyradiff::Tensor convert_tensor<float>(torch::Tensor tensor);
template lyradiff::Tensor convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template lyradiff::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template lyradiff::Tensor convert_tensor<int>(torch::Tensor tensor);
template lyradiff::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor);
template lyradiff::Tensor convert_tensor<unsigned int>(torch::Tensor tensor);
template lyradiff::Tensor convert_tensor<bool>(torch::Tensor tensor);

template<typename T>
lyradiff::Tensor convert_tensor(torch::Tensor tensor, lyradiff::MemoryType memory_type)
{
    return lyradiff::Tensor{memory_type, lyradiff::getTensorType<T>(), convert_shape(tensor), get_ptr<T>(tensor)};
}

template lyradiff::Tensor convert_tensor<int8_t>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
template lyradiff::Tensor convert_tensor<float>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
template lyradiff::Tensor convert_tensor<half>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
#ifdef ENABLE_BF16
template lyradiff::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
#endif
template lyradiff::Tensor convert_tensor<int>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
template lyradiff::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
template lyradiff::Tensor convert_tensor<unsigned int>(torch::Tensor tensor, lyradiff::MemoryType memory_type);
template lyradiff::Tensor convert_tensor<bool>(torch::Tensor tensor, lyradiff::MemoryType memory_type);

size_t sizeBytes(torch::Tensor tensor)
{
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

lyradiff::FtCudaDataType parse_model_dtype_str(const std::string& model_dtype)
{
    lyradiff::FtCudaDataType model_file_type;
    if (model_dtype == "fp32") {
        model_file_type = lyradiff::FtCudaDataType::FP32;
    }
    else if (model_dtype == "fp16") {
        model_file_type = lyradiff::FtCudaDataType::FP16;
    }
    else {
        throw "wrong model_dtype";
    }
    return model_file_type;
}

void insertParamsToMap(lyradiff::TensorMap& tensor_map, torch::optional<torch::Dict<std::string, double>> scale_params)
{
    if (!scale_params.has_value())
        return;
    auto  param_map_ptr = tensor_map.getParamsMapPtr();
    auto& dict          = scale_params.value();
    for (auto iter = dict.begin(); iter != dict.end(); ++iter) {
        param_map_ptr->insert({iter->key(), iter->value()});
    }
}

}  // namespace torch_ext
