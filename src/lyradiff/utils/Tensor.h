#pragma once

#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/common_name_def.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/string_utils.h"

#include "stdlib.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace lyradiff {

typedef enum datatype_enum {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID,
} DataType;

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return TYPE_FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return TYPE_FP16;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, const __nv_bfloat16>::value) {
        return TYPE_BF16;
    }
#endif
#ifdef ENABLE_FP8
    else if (std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, const __nv_fp8_e4m3>::value) {
        return TYPE_FP8_E4M3;
    }
#endif
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return TYPE_INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return TYPE_INT8;
    }
    else if (std::is_same<T, uint>::value || std::is_same<T, const uint>::value) {
        return TYPE_UINT32;
    }
    else if (std::is_same<T, unsigned long long int>::value || std::is_same<T, const unsigned long long int>::value) {
        return TYPE_UINT64;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return TYPE_BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return TYPE_BYTES;
    }
    else {
        return TYPE_INVALID;
    }
}

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    const MemoryType          where;
    const DataType            type;
    const std::vector<size_t> shape;
    const void*               data;  // TODO(bhseuh) modify from const void* to void* const
    const std::vector<size_t> offsets = std::vector<size_t>{};

    Tensor();
    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data);
    Tensor(const MemoryType          _where,
           const DataType            _type,
           const std::vector<size_t> _shape,
           const void*               _data,
           const std::vector<size_t> _offset);
    Tensor(Tensor const& other);

    Tensor operator=(Tensor const& other);

    size_t size() const;
    size_t sizeBytes() const;

    std::string getShapeStr()
    {
        std::string s = "(";
        for (int i = 0; i < shape.size(); i++) {
            s += std::to_string(shape[i]);
            if (i != shape.size() - 1)
                s += ", ";
        }
        s += ")";
        return s;
    }

    std::string whereToString() const;
    std::string toString() const;
    std::string getNumpyTypeDesc(DataType type) const;

    void          saveNpy(const std::string& filename) const;
    static Tensor loadNpy(const std::string& npy_file, const MemoryType where);

    static Tensor loadTorchTensor(const std::string& torch_file, const MemoryType where);

    static DataType typeFromNumpyDesc(std::string type);
    static size_t   getTypeSize(DataType type);

    template<typename T>
    inline T getVal(size_t index) const
    {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        LYRA_CHECK(where == MEMORY_CPU);
        LYRA_CHECK(data != nullptr);
        LYRA_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");

        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return ((T*)data)[index];
    }

    template<typename T>
    inline T getVal() const
    {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return getVal<T>(0);
    }

    template<typename T>
    inline T* getPtr() const
    {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getPtr with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return (T*)data;
    }

    inline void* getPtrWithOffset(size_t offset) const
    {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (data == nullptr) {
            return (void*)data;
        }
        else {
            LYRA_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return (void*)((char*)data + offset * Tensor::getTypeSize(type));
        }
    }

    template<typename T>
    inline T* getPtrWithOffset(size_t offset) const
    {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        if (data == nullptr) {
            return (T*)data;
        }
        else {
            LYRA_CHECK_WITH_INFO(offset < size(),
                                 fmtstr("offset (%lu) is larger than buffer size (%lu)", offset, size()));
            return ((T*)data) + offset;
        }
    }

    template<typename T>
    T max() const
    {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        LYRA_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        LYRA_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                             "max() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t max_idx = 0;
        T      max_val = getVal<T>(max_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        return max_val;
    }

    template<typename T>
    T min() const
    {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        LYRA_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        LYRA_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                             "min() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t min_idx = 0;
        T      min_val = getVal<T>(min_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val < min_val) {
                min_idx = i;
                min_val = val;
            }
        }
        return min_val;
    }

    template<typename T>
    T any(T val) const
    {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        LYRA_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        LYRA_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                             "any() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) == val) {
                return true;
            }
        }
        return false;
    }

    template<typename T>
    T all(T val) const
    {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        LYRA_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        LYRA_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                             "all() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) != val) {
                return false;
            }
        }
        return true;
    }

    void updateShape(size_t idx, size_t val)
    {
        // TODO: find a better way to update the shape
        std::vector<size_t>& shape_ref = const_cast<std::vector<size_t>&>(shape);
        shape_ref[idx]                 = val;
    }

    Tensor slice(std::vector<size_t> shape, size_t offset = 0) const;

    template<typename T_OUT, typename T_IN>
    Tensor convert_gpu_dtype(cudaStream_t stream = cudaStreamDefault) const;

private:
    static void parseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data);
    static int  parseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, std::vector<size_t>& shape);
};

class LyraDiffContext;

class TensorMap {
protected:
    std::unordered_map<std::string, Tensor> tensor_map_;
    std::unordered_map<std::string, float>* param_map_ptr_;

    inline bool isValid(const Tensor& tensor)
    {
        return tensor.size() > 0 && tensor.data != nullptr;
    }

public:
    LyraDiffContext* context_ = nullptr;

public:
    TensorMap() = default;
    TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map);
    TensorMap(const std::vector<Tensor>& tensor_map);
    TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
    ~TensorMap();

    void       setContext(LyraDiffContext* ctx);
    TensorMap& setContextThis(LyraDiffContext* ctx);
    TensorMap& setContextThis(const TensorMap* mm);

    inline void reinit(const std::unordered_map<std::string, Tensor>& tensor_map)
    {
        tensor_map_ = tensor_map;
    }

    inline size_t size() const
    {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const
    {
        FT_LOG_DEBUG("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const std::string& key) const
    {
        FT_LOG_DEBUG("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
        return tensor_map_.find(key) != tensor_map_.end() && tensor_map_.at(key).type != TYPE_INVALID;
    }

    std::vector<std::string> keys() const;

    inline void insert(const std::string& key, const Tensor& value)
    {
        LYRA_CHECK_WITH_INFO(!isExist(key), fmtstr("Duplicated key %s", key.c_str()));
        LYRA_CHECK_WITH_INFO(isValid(value),
                             fmtstr("A none tensor or nullptr is not allowed (key is %s)", key.c_str()));
        tensor_map_.insert({key, value});
    }

    inline void insertIfValid(const std::string& key, const Tensor& value)
    {
        if (isValid(value)) {
            insert({key, value});
        }
    }

    inline void insert(std::pair<std::string, Tensor> p)
    {
        tensor_map_.insert(p);
    }

    TensorMap& merge(TensorMap& m);

    TensorMap subMap(const std::vector<std::string>& vec_keys) const;

    // prevent converting int or size_t to string automatically
    Tensor at(int tmp)    = delete;
    Tensor at(size_t tmp) = delete;

    inline Tensor& at(const std::string& key)
    {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor at(const std::string& key) const
    {
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor& at(const std::string& key, Tensor& default_tensor)
    {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor& default_tensor) const
    {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor& at(const std::string& key, Tensor&& default_tensor)
    {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor&& default_tensor) const
    {
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    template<typename T>
    inline T getVal(const std::string& key) const
    {
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>();
    }

    template<typename T>
    inline T getVal(const std::string& key, T default_value) const
    {
        if (isExist(key)) {
            return tensor_map_.at(key).getVal<T>();
        }
        return default_value;
    }

    template<typename T>
    inline T getValWithOffset(const std::string& key, size_t index) const
    {
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>(index);
    }

    template<typename T>
    inline T getValWithOffset(const std::string& key, size_t index, T default_value) const
    {
        if (isExist(key)) {
            return tensor_map_.at(key).getVal<T>(index);
        }
        return default_value;
    }

    template<typename T>
    inline T* getPtr(const std::string& key) const
    {
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtr<T>();
    }

    template<typename T>
    inline T* getPtr(const std::string& key, T* default_ptr) const
    {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtr<T>();
        }
        return default_ptr;
    }

    template<typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index) const
    {
        LYRA_CHECK_WITH_INFO(isExist(key),
                             fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                    key.c_str(),
                                    vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtrWithOffset<T>(index);
    }

    template<typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index, T* default_ptr) const
    {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtrWithOffset<T>(index);
        }
        return default_ptr;
    }

    inline std::unordered_map<std::string, Tensor> getMap() const
    {
        return tensor_map_;
    }

    inline std::unordered_map<std::string, float>* getParamsMapPtr() const
    {
        return param_map_ptr_;
    }

    inline std::unordered_map<std::string, Tensor>::iterator begin()
    {
        return tensor_map_.begin();
    }

    inline std::unordered_map<std::string, Tensor>::iterator end()
    {
        return tensor_map_.end();
    }

    std::string      toString();
    static TensorMap fromNpyFolder(const std::string& base_folder);
    void             saveNpy(const std::string& base_folder);
};

extern bool is_debug_memroy_utils_s3diff;

class LoraWeightContainer {
public:
    std::unordered_map<std::string, void*> map_lora_alpha_weights;
    std::unordered_map<std::string, void*> map_lora_beta_weights;

    std::unordered_map<void*, void*> map_default_weight_to_lora_alpha;
    std::unordered_map<void*, void*> map_default_weight_to_lora_beta;

    // 输入的unet_embed de_mods (for s3diff)
    std::unordered_map<std::string, void*> map_de_mods;
    static std::vector<std::string>        all_de_mod_prefixs;

public:
    void*              get_lora_weight(const std::string& name, bool is_alpha = true);
    void*              get_lora_weight_by_ptr(void* src_ptr, bool is_alpha = true);
    void               add_lora_weight(const std::string& name, void* ptr, bool is_alpha = true);
    void               set_map_de_mods(const std::string& name, void* ptr);
    static std::string get_de_mod_prefix(const std::string& name);
    void*              get_de_mod(const std::string& name);
};

class WeightLoaderManager {
public:
    // 当前 load 的 module: unet / vae / controlnet / etc
    std::string cur_load_module;
    // 不同 module 对应的 weight_map
    std::unordered_map<std::string, std::unordered_map<std::string, void*>> map_module_weights;
    std::unordered_map<std::string, std::unordered_map<void*, std::string>> map_module_rev_weights;
    // 获取输入 weight_map reverse_map
    std::unordered_map<void*, std::string> map_weights_reverse;

    // for s3diff 的 lora container
    // LoraWeightContainer lora_container;
    std::unordered_map<std::string, LoraWeightContainer> map_lora_container;
    // 当前 context
    LyraDiffContext* ctx;
    // // cublasMMWrapper
    // cublasMMWrapper* cublas_wrapper;

public:
    // 设置当前的module
    void set_cur_load_module(const std::string& cur_module);
    // 设置 reverse weight map（文件读取的 weight ptr -> name
    void set_map_weights_reverse(const std::unordered_map<std::string, void*> map_weights);
    // 添加 lyra weight ptr
    void add_weight(const std::string& name, void* ptr);
    // 做 memcpy
    void doCudaMemcpy(void* dst, void* src, size_t elem_cnt, cudaMemcpyKind kind);
};
extern WeightLoaderManager*            weight_loader_manager_glob;
extern Allocator<AllocatorType::CUDA>* allocator_glob;

}  // namespace lyradiff
