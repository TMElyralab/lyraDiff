#pragma once
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/context.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>
#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                                                         \
    CHECK_CPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_OPTIONAL_INPUT(x, st)                                                                                    \
    if (x.has_value()) {                                                                                               \
        CHECK_INPUT(x.value(), st);                                                                                    \
    }
#define CHECK_OPTIONAL_CPU_INPUT(x, st)                                                                                \
    if (x.has_value()) {                                                                                               \
        CHECK_CPU_INPUT(x.value(), st);                                                                                \
    }
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace torch_ext {

template<typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}

std::vector<size_t> convert_shape(torch::Tensor tensor);

template<typename T>
lyradiff::Tensor convert_tensor(torch::Tensor tensor);

template<typename T>
lyradiff::Tensor convert_tensor(torch::Tensor tensor, lyradiff::MemoryType memory_type);

size_t sizeBytes(torch::Tensor tensor);

lyradiff::FtCudaDataType parse_model_dtype_str(const std::string& model_dtype);

template<typename T>
void insertTorchTensorToMap(lyradiff::TensorMap&              tensor_map,
                            const std::string&              name,
                            torch::optional<torch::Tensor>& tensor)
{
    if (tensor.has_value()) {
        tensor_map.insert(name, convert_tensor<T>(tensor.value()));
    }
    else {
        tensor_map.insert(make_pair(name, lyradiff::Tensor()));
    }
}

template<typename T>
void insertTorchTensorMapToMapContext(lyradiff::TensorMap&                                       tensor_map,
                                      torch::optional<torch::Dict<std::string, torch::Tensor>> map_extra_tensors)
{
    if (!map_extra_tensors.has_value())
        return;
    for (auto iter = map_extra_tensors.value().begin(); iter != map_extra_tensors.value().end(); iter++) {
        auto name   = iter->key();
        auto tensor = iter->value();
        tensor_map.context_->insert(std::make_pair(name, convert_tensor<T>(tensor)));
    }
}

template<typename T>
static void loadDataFromBin(std::vector<T>& buffer, size_t size, std::string filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        throw "file %s cannot be opened, loading model fails! \n", filename.c_str();
        return;
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    in.read((char*)buffer.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    // cout << "in_get_size: " << in_get_size << endl;
    // cout << "loaded_data_size: " << loaded_data_size << endl;
    if (in_get_size != loaded_data_size) {
        throw std::runtime_error("Wrong Size for bin file load");
    }
    in.close();
}

template static void loadDataFromBin(std::vector<half>& buffer, size_t size, std::string filename);
template static void loadDataFromBin(std::vector<float>& buffer, size_t size, std::string filename);

template<typename T>
torch::Tensor getTensorFromBin(std::string bin_path, size_t size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    if (std::is_same<T, float>()) {
        options = torch::TensorOptions().dtype(torch::kFloat32);
    }

    std::vector<T> buffer(size);
    loadDataFromBin<T>(buffer, size, bin_path);
    torch::Tensor ret_tensor = torch::from_blob(buffer.data(), {size}, options).clone();
    return ret_tensor;
}

void insertParamsToMap(lyradiff::TensorMap& tensor_map, torch::optional<torch::Dict<std::string, double>> scale_params);

}  // namespace torch_ext
