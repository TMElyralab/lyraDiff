#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/context.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/string_utils.h"
#include "src/lyradiff/utils/tensor_helper.h"

#include "stdlib.h"
#include <cuda_fp16.h>
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

Tensor::Tensor():
    // a none tensor.
    where(MEMORY_CPU), type(TYPE_INVALID), shape({}), data(nullptr), offsets({})  // only a record to record offset
{
}

TensorMap& TensorMap::merge(TensorMap& m)
{
    for (auto iter : m.getMap()) {
        this->insert(iter);
    }
    setContext(m.context_);
    return *this;
}

TensorMap TensorMap::subMap(const std::vector<std::string>& vec_keys) const
{
    TensorMap sub_map;
    for (auto key : vec_keys) {
        if (tensor_map_.find(key) != tensor_map_.end()) {
            sub_map.insert(make_pair(key, tensor_map_.at(key)));
        }
    }
    sub_map.setContext(this->context_);
    return sub_map;
}

void TensorMap::setContext(LyraDiffContext* ctx)
{
    context_       = ctx;
    param_map_ptr_ = &context_->map_scale_params;
}

TensorMap& TensorMap::setContextThis(LyraDiffContext* ctx)
{
    setContext(ctx);
    return *this;
}

TensorMap& TensorMap::setContextThis(const TensorMap* mm)
{
    if (mm->context_ != nullptr) {
        setContext(mm->context_);
    }
    return *this;
}

Tensor::Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data):
    where(_where), type(_type), shape(_shape), data(_data)
{
}

Tensor::Tensor(const MemoryType          _where,
               const DataType            _type,
               const std::vector<size_t> _shape,
               const void*               _data,
               const std::vector<size_t> _offset):
    where(_where), type(_type), shape(_shape), data(_data), offsets(_offset)
{
}

Tensor::Tensor(Tensor const& other):
    where(other.where), type(other.type), shape(other.shape), data(other.data), offsets(other.offsets)
{
}

Tensor Tensor::operator=(Tensor const& other)
{
    // this->
    return other;
}

void Tensor::parseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data)
{
    const char magic[]                   = "\x93"
                                           "NUMPY";
    char       magic_test[sizeof(magic)] = "\0";

    size_t n_elems = fread((void*)magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
    if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
        throw std::runtime_error("Could read magic token in NPY file");
    }

    uint8_t npy_major = 0;
    uint8_t npy_minor = 0;
    n_elems           = fread((void*)&npy_major, sizeof(uint8_t), 1, f_ptr);
    n_elems += fread((void*)&npy_minor, sizeof(uint8_t), 1, f_ptr);

    if (npy_major == 1) {
        uint16_t header_len_u16 = 0;
        n_elems                 = fread((void*)&header_len_u16, sizeof(uint16_t), 1, f_ptr);
        header_len              = header_len_u16;
    }
    else if (npy_major == 2) {
        uint32_t header_len_u32 = 0;
        n_elems                 = fread((void*)&header_len_u32, sizeof(uint32_t), 1, f_ptr);
        header_len              = header_len_u32;
    }
    else {
        throw std::runtime_error("Unsupported npy version: " + std::to_string(npy_major));
    }

    start_data = 8 + 2 * npy_major + header_len;
}

int Tensor::parseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, std::vector<size_t>& shape)
{
    char*  header_c = (char*)malloc(header_len * sizeof(char));
    size_t n_elems  = fread((void*)header_c, sizeof(char), header_len, f_ptr);
    if (n_elems != header_len) {
        free(header_c);
        return -1;
    }
    std::string header(header_c, header_len);
    free(header_c);

    size_t start, end;
    start = header.find("'descr'") + 7;
    start = header.find("'", start);
    end   = header.find("'", start + 1);
    type  = typeFromNumpyDesc(header.substr(start + 1, end - start - 1));

    start = header.find("'fortran_order'") + 15;
    start = header.find(":", start);
    end   = header.find(",", start + 1);
    if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
        throw std::runtime_error("Unsupported value for fortran_order while reading npy file");
    }

    start = header.find("'shape'") + 7;
    start = header.find("(", start);
    end   = header.find(")", start + 1);

    std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
    std::string        token;

    shape.clear();
    while (std::getline(shape_stream, token, ',')) {
        if (token.find_first_not_of(' ') == std::string::npos) {
            break;
        }
        shape.push_back(std::stoul(token));
    }

    return 0;
}

Tensor Tensor::loadNpy(const std::string& npy_file, const MemoryType where)
{
    DataType            type;
    std::vector<size_t> shape;

    FILE* f_ptr = fopen(npy_file.c_str(), "rb");
    if (f_ptr == nullptr) {
        throw std::runtime_error("Could not open file " + npy_file);
    }
    uint32_t header_len, start_data;
    parseNpyIntro(f_ptr, header_len, start_data);
    parseNpyHeader(f_ptr, header_len, type, shape);

    const size_t size     = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    void*        data_cpu = malloc(size * Tensor::getTypeSize(type));
    void*        data     = data_cpu;

    size_t n_elems = fread(data_cpu, Tensor::getTypeSize(type), size, f_ptr);
    LYRA_CHECK_WITH_INFO(n_elems == size, "reading tensor failed");
    if (where == MEMORY_GPU) {
        cudaMalloc(&data, size * Tensor::getTypeSize(type));
        cudaMemcpy(data, data_cpu, size * Tensor::getTypeSize(type), cudaMemcpyHostToDevice);
        free(data_cpu);
    }

    fclose(f_ptr);
    return Tensor(where, type, shape, data);
}

size_t Tensor::size() const
{
    if (data == nullptr || shape.size() == 0) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::sizeBytes() const
{
    return size() * Tensor::getTypeSize(type);
}

std::string Tensor::whereToString() const
{
    static const std::unordered_map<MemoryType, std::string> mem_to_string{
        {MEMORY_CPU, "CPU"}, {MEMORY_CPU_PINNED, "CPU_PINNED"}, {MEMORY_GPU, "GPU"}};
    return mem_to_string.at(where);
}

std::string Tensor::toString() const
{
    std::string memtype_str = whereToString();

    static const std::unordered_map<DataType, std::string> type_to_string{
        {TYPE_BOOL, "BOOL"},
        {TYPE_UINT8, "UINT8"},
        {TYPE_UINT16, "UINT16"},
        {TYPE_UINT32, "UINT32"},
        {TYPE_UINT64, "UINT64"},
        {TYPE_INT8, "INT8"},
        {TYPE_INT16, "INT16"},
        {TYPE_INT32, "INT32"},
        {TYPE_INT64, "INT64"},
        {TYPE_BF16, "BF16"},
        {TYPE_FP16, "FP16"},
        {TYPE_FP32, "FP32"},
        {TYPE_FP64, "FP64"},
        {TYPE_BYTES, "BYTES"},
        {TYPE_INVALID, "INVALID"},
        {TYPE_FP8_E4M3, "E4M3"},
        {TYPE_VOID, "VOID"},
    };
    return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                  memtype_str.c_str(),
                  type_to_string.at(type).c_str(),
                  vec2str(shape).c_str(),
                  data);
}

DataType Tensor::typeFromNumpyDesc(std::string type)
{
    size_t pos = type.find_first_of("=<>|");
    if (pos != std::string::npos)
        type.replace(pos, 1, "");

    static const std::unordered_map<std::string, DataType> type_map{{"?", TYPE_BOOL},
                                                                    {"b", TYPE_BYTES},
                                                                    {"u1", TYPE_UINT8},
                                                                    {"u2", TYPE_UINT16},
                                                                    {"u4", TYPE_UINT32},
                                                                    {"u8", TYPE_UINT64},
                                                                    {"i1", TYPE_INT8},
                                                                    {"i2", TYPE_INT16},
                                                                    {"i4", TYPE_INT32},
                                                                    {"i8", TYPE_INT64},
                                                                    {"f2", TYPE_FP16},
                                                                    {"f4", TYPE_FP32},
                                                                    {"f8", TYPE_FP64}};
    return type_map.at(type);
}

size_t Tensor::getTypeSize(DataType type)
{
    static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)},
                                                               {TYPE_BYTES, sizeof(char)},
                                                               {TYPE_UINT8, sizeof(uint8_t)},
                                                               {TYPE_UINT16, sizeof(uint16_t)},
                                                               {TYPE_UINT32, sizeof(uint32_t)},
                                                               {TYPE_UINT64, sizeof(uint64_t)},
                                                               {TYPE_INT8, sizeof(int8_t)},
                                                               {TYPE_INT16, sizeof(int16_t)},
                                                               {TYPE_INT32, sizeof(int32_t)},
                                                               {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
                                                               {TYPE_BF16, sizeof(float)}, 
#endif
#ifdef ENABLE_FP8
                                                               {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
#endif
                                                               {TYPE_FP16, sizeof(half)},
                                                               {TYPE_FP32, sizeof(float)},
                                                               {TYPE_FP64, sizeof(double)}};
    return type_map.at(type);
}

std::string Tensor::getNumpyTypeDesc(DataType type) const
{
    static const std::unordered_map<DataType, std::string> type_map{
        {TYPE_INVALID, "x"},
        {TYPE_BOOL, "?"},
        {TYPE_BYTES, "b"},
        {TYPE_UINT8, "u1"},
        {TYPE_UINT16, "u2"},
        {TYPE_UINT32, "u4"},
        {TYPE_UINT64, "u8"},
        {TYPE_INT8, "i1"},
        {TYPE_INT16, "i2"},
        {TYPE_INT32, "i4"},
        {TYPE_INT64, "i8"},
        {TYPE_FP16, "f2"},
        {TYPE_FP32, "f4"},
        {TYPE_BF16, "f4"}, 
        {TYPE_FP64, "f8"}};

    if (type == TYPE_BF16) {
        FT_LOG_WARNING("getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
                       "support bfloat16 as of now, it will be properly extended if numpy supports. "
                       "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

void Tensor::saveNpy(const std::string& filename) const
{
    // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
    void*  cpu_data     = (void*)data;
    bool   is_data_temp = false;
    size_t tensor_size  = size();
    if (where == MemoryType::MEMORY_GPU) {
        cpu_data     = malloc(tensor_size * Tensor::getTypeSize(type));
        is_data_temp = true;
        cudaDeviceSynchronize();
        if (type == DataType::TYPE_BF16) {
            float* tmp_data;
            cudaMalloc(&tmp_data, tensor_size * Tensor::getTypeSize(type));
            invokeCudaD2DcpyConvert(tmp_data, (__nv_bfloat16*)data, tensor_size);
            cudaMemcpy(cpu_data, tmp_data, tensor_size * Tensor::getTypeSize(type), cudaMemcpyDeviceToHost);
            deviceFree(tmp_data);
        }
        else {
            cudaMemcpy(cpu_data, data, tensor_size * Tensor::getTypeSize(type), cudaMemcpyDeviceToHost);
        }
    }

    const char    magic[]   = "\x93"
                              "NUMPY";
    const uint8_t npy_major = 1;
    const uint8_t npy_minor = 0;

    std::stringstream header_stream;
    header_stream << "{'descr': '" << getNumpyTypeDesc(type) << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header_stream << shape[i];
        if (i + 1 < shape.size() || shape.size() == 1) {
            header_stream << ", ";
        }
    }
    header_stream << ")}";
    int base_length = 6 + 4 + header_stream.str().size();
    int pad_length  = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
    for (int i = 0; i < pad_length - base_length; ++i) {
        header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
    }
    std::string    header     = header_stream.str();
    const uint16_t header_len = header.size();

    FILE* f_ptr = fopen(filename.c_str(), "wb");
    LYRA_CHECK_WITH_INFO(f_ptr != nullptr, fmtstr("Unable to open %s for writing.\n", filename.c_str()));

    fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
    fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
    fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
    fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
    fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
    fwrite(cpu_data, Tensor::getTypeSize(type), tensor_size, f_ptr);

    fclose(f_ptr);

    if (is_data_temp) {
        free(cpu_data);
    }
}

Tensor Tensor::slice(std::vector<size_t> shape, size_t offset) const
{
    if (this->data != nullptr) {
        size_t n_elts        = this->size();
        size_t n_sliced_elts = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
        LYRA_CHECK_WITH_INFO(
            n_sliced_elts + offset <= n_elts,
            fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                   n_sliced_elts + offset,
                   n_elts));
    }
    return Tensor(this->where, this->type, shape, this->getPtrWithOffset(offset));
}

template<typename T_OUT, typename T_IN>
Tensor Tensor::convert_gpu_dtype(cudaStream_t stream) const
{
    size_t size = this->size();
    T_OUT* dst;
    cudaMallocAsync(&dst, size * sizeof(T_OUT), stream);

    invokeTensorD2DConvert<T_OUT, T_IN>(dst, this->getPtr<T_IN>(), size, stream);
    cudaStreamSynchronize(stream);

    return Tensor(this->where, getTensorType<T_OUT>(), this->shape, dst);
}

template Tensor Tensor::convert_gpu_dtype<half, float>(cudaStream_t stream) const;
template Tensor Tensor::convert_gpu_dtype<float, half>(cudaStream_t stream) const;

// TensorMap
TensorMap::TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map)
{
    for (auto& kv : tensor_map) {
        if (isValid(kv.second)) {
            insert(kv.first, kv.second);
        }
        else {
            FT_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str()));
        }
    }
}

TensorMap::TensorMap(const std::vector<Tensor>& tensor_map)
{
    for (size_t i = 0; i < tensor_map.size(); i++) {
        insert(std::to_string(i), tensor_map[i]);
    }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map)
{
    for (auto& pair : tensor_map) {
        if (isValid(pair.second)) {
            insert(pair.first, pair.second);
        }
        else {
            FT_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
        }
    }
}

TensorMap::~TensorMap()
{
    tensor_map_.clear();
}

std::vector<std::string> TensorMap::keys() const
{
    std::vector<std::string> key_names;
    for (auto& kv : tensor_map_) {
        key_names.push_back(kv.first);
    }
    return key_names;
}

std::string TensorMap::toString()
{
    std::stringstream ss;
    ss << "{";
    std::vector<std::string> key_names = keys();
    for (size_t i = 0; i < tensor_map_.size(); ++i) {
        ss << key_names[i] << ": " << at(key_names[i]).toString();
        if (i < tensor_map_.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}

TensorMap TensorMap::fromNpyFolder(const std::string& base_folder)
{
    DIR* dir_p = opendir(base_folder.c_str());
    LYRA_CHECK_WITH_INFO(dir_p != nullptr, fmtstr("Could not open folder %s. ", base_folder.c_str()));
    struct dirent* dp;

    TensorMap ret_tensor;
    while ((dp = readdir(dir_p)) != nullptr) {
        std::string filename(dp->d_name);
        size_t      len = filename.length();
        if (len < 4 || filename.compare(len - 4, 4, ".npy")) {
            continue;
        }

        size_t pos = filename.find('-');
        LYRA_CHECK_WITH_INFO(pos != std::string::npos, fmtstr("Invalid filename: %s\n", filename.c_str()));

        MemoryType where;
        if (filename.compare(0, pos, "GPU") == 0) {
            where = MEMORY_GPU;
        }
        else if (filename.compare(0, pos, "CPU") == 0) {
            where = MEMORY_CPU;
        }
        else if (filename.compare(0, pos, "CPU_PINNED") == 0) {
            where = MEMORY_CPU_PINNED;
        }
        else {
            LYRA_CHECK_WITH_INFO(false, fmtstr("Invalid filename: %s\n", filename.c_str()));
        }
        std::string key = filename.substr(pos + 1, len - pos - 5);

        ret_tensor.tensor_map_.insert({key, Tensor::loadNpy(base_folder + "/" + filename, where)});
    }

    closedir(dir_p);

    return ret_tensor;
}

void TensorMap::saveNpy(const std::string& base_folder)
{
    mode_t mode_0755 = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
    int    ret       = mkdir(base_folder.c_str(), mode_0755);
    LYRA_CHECK_WITH_INFO(ret == 0 || errno == EEXIST, fmtstr("Could not create folder %s.\n", base_folder.c_str()));

    for (const auto& item : tensor_map_) {
        item.second.saveNpy(base_folder + "/" + item.second.whereToString() + "-" + item.first + ".npy");
    }
}

bool                            is_debug_memroy_utils_s3diff = false;
WeightLoaderManager*            weight_loader_manager_glob   = new WeightLoaderManager();
Allocator<AllocatorType::CUDA>* allocator_glob               = nullptr;

// ============== LoraWeightContainer ===============
void* LoraWeightContainer::get_lora_weight(const std::string& name, bool is_alpha)
{
    auto& map_lora = is_alpha ? map_lora_alpha_weights : map_lora_beta_weights;
    if (map_lora.find(name) == map_lora.end()) {
        PRINTF_S3DIFF("bug... can't find lora: %s, mapsize: %d\n", name.c_str(), map_lora.size());
        return nullptr;
    }
    return map_lora.at(name);
}
void* LoraWeightContainer::get_lora_weight_by_ptr(void* src_ptr, bool is_alpha)
{
    auto& map_lora = is_alpha ? map_default_weight_to_lora_alpha : map_default_weight_to_lora_beta;
    if (map_lora.find(src_ptr) == map_lora.end()) {
        PRINTF_S3DIFF("bug... cant find lora by ptr\n");
        return nullptr;
    }
    return map_lora.at(src_ptr);
}

void LoraWeightContainer::add_lora_weight(const std::string& name, void* ptr, bool is_alpha)
{
    auto& map_lora       = is_alpha ? map_lora_alpha_weights : map_lora_beta_weights;
    map_lora[name]       = ptr;
    auto cur_load_module = weight_loader_manager_glob->cur_load_module;
    if (weight_loader_manager_glob->map_module_weights[cur_load_module].find(name)
        == weight_loader_manager_glob->map_module_weights[cur_load_module].end()) {
        PRINTF_S3DIFF("add_lora_weight..... cant find: %s, size: %d\n",
                      name.c_str(),
                      weight_loader_manager_glob->map_module_weights[cur_load_module].size());
    }
}

void LoraWeightContainer::set_map_de_mods(const std::string& name, void* ptr)
{
    PRINTF_S3DIFF("set_map_de_mods: %s, %p\n", name.c_str(), ptr);
    map_de_mods[name] = ptr;
}

std::vector<std::string> LoraWeightContainer::all_de_mod_prefixs = {"up_blocks.", "down_blocks.", "mid_block"};

std::string LoraWeightContainer::get_de_mod_prefix(const std::string& name)
{
    auto pos = name.find(".");
    if (pos == name.npos)
        return "other";

    std::string prefix = "other";
    for (auto& avail_name : all_de_mod_prefixs) {
        int pos = name.find(avail_name);
        if (pos == name.npos) {
            continue;
        }
        int pos_add = name.find("mid_block.") == name.npos ? 1 : 0;
        prefix      = name.substr(0, pos + avail_name.length() + pos_add);
        break;
    }
    if (prefix != "other") {
        if (name.find("attn1.to_q") != name.npos) {
            prefix = prefix + "_attn1_qkv";
        }
        else if (name.find("attn2.to_k") != name.npos) {
            prefix = prefix + "_attn2_kv";
        }
        else if (name.find("attentions.0.to_q") != name.npos) {
            prefix = prefix + "_attn1_qkv";
        }
    }
    return prefix;
}

void* LoraWeightContainer::get_de_mod(const std::string& name)
{
    auto prefix = get_de_mod_prefix(name);
    PRINTF_S3DIFF("demod prefix: %s, map_de_mods size: %d\n", prefix.c_str(), map_de_mods.size());
    // for(auto iter: map_de_mods){
    //     printf("%s\n", iter.first.c_str());
    // }
    if (map_de_mods.find(prefix) == map_de_mods.end())
        return nullptr;
    return map_de_mods.at(prefix);
}

// ============== WeightLoaderManager ===============
void WeightLoaderManager::set_cur_load_module(const std::string& cur_module)
{
    cur_load_module = cur_module;
}
void WeightLoaderManager::set_map_weights_reverse(const std::unordered_map<std::string, void*> map_weights)
{
    map_weights_reverse.clear();
    for (auto iter : map_weights) {
        map_weights_reverse[iter.second] = iter.first;
    }
}

void WeightLoaderManager::add_weight(const std::string& name, void* ptr)
{
    map_module_weights[cur_load_module][name]    = ptr;
    map_module_rev_weights[cur_load_module][ptr] = name;
    PRINTF_S3DIFF("cur module weightcnt: %d, module: %s, name: %s\n",
                  map_module_weights[cur_load_module].size(),
                  cur_load_module.c_str(),
                  name.c_str());
}

void WeightLoaderManager::doCudaMemcpy(void* dst, void* src, size_t elem_cnt, cudaMemcpyKind kind)
{
    cudaMemcpy(dst, src, elem_cnt, kind);

    auto weight_name = map_weights_reverse[src];
    add_weight(weight_name, dst);
}

}  // namespace lyradiff
