#pragma once
#include "IFBaseWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/allocator.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace lyradiff {

template<typename T>
class LoraWeight {
private:
    std::vector<size_t> weight_shape_;
    T*                  weight_     = nullptr;
    bool                loaded_lora = false;

    size_t get_weight_size()
    {
        return std::accumulate(weight_shape_.begin(), weight_shape_.end(), (size_t)1, std::multiplies<size_t>());
    }

public:
    // for non-int8 weight
    LoraWeight() {}
    LoraWeight(const std::vector<size_t> weight_shape, T* weight): weight_shape_(weight_shape), weight_(weight) {}

    virtual void load_lora(const T* lora_weight, const float lora_scale, IAllocator* allocator)
    {
        size_t lora_size = get_weight_size();
        invokeLoadLora<T>(weight_, lora_weight, lora_size, lora_scale);
    }

    virtual void clear_lora(IAllocator* allocator) {}
};

template<typename T>
class LoraWeightV2: public LoraWeight<T> {
private:
    std::vector<size_t> weight_shape_;
    T*                  weight_     = nullptr;
    T*                  weight_h_   = nullptr;
    bool                loaded_lora = false;

    size_t get_weight_size()
    {
        return std::accumulate(weight_shape_.begin(), weight_shape_.end(), (size_t)1, std::multiplies<size_t>());
    }

public:
    // for non-int8 weight
    LoraWeightV2() {}
    LoraWeightV2(const std::vector<size_t> weight_shape, T* weight, T* weight_h):
        weight_shape_(weight_shape), weight_(weight), weight_h_(weight_h)
    {
    }

    virtual void load_lora(const T* lora_weight, const float lora_scale, IAllocator* allocator) override
    {
        size_t lora_size = get_weight_size();
        cudaMemcpyAsync(weight_, weight_h_, sizeof(T) * lora_size, cudaMemcpyHostToDevice, cudaStreamDefault);
        invokeLoadLora<T>(weight_, lora_weight, lora_size, lora_scale);
        loaded_lora = true;
    }

    virtual void clear_lora(IAllocator* allocator) override
    {
        if (!loaded_lora) {
            return;
        }
        size_t lora_size = get_weight_size();
        cudaMemcpyAsync(weight_, weight_h_, sizeof(T) * lora_size, cudaMemcpyHostToDevice, cudaStreamDefault);
    }
};

template<typename T>
class FP8LoraWeight: public LoraWeight<T> {
private:
    std::vector<size_t> weight_shape_;
    __nv_fp8_e4m3*      weight_       = nullptr;
    float*              weight_scale_ = nullptr;
    T*                  weight_h_     = nullptr;
    bool                loaded_lora   = false;

    size_t get_weight_size()
    {
        return std::accumulate(weight_shape_.begin(), weight_shape_.end(), (size_t)1, std::multiplies<size_t>());
    }

public:
    // for non-int8 weight
    FP8LoraWeight() {}
    FP8LoraWeight(const std::vector<size_t> weight_shape, __nv_fp8_e4m3* weight, float* weight_scale, T* weight_h):
        weight_shape_(weight_shape), weight_(weight), weight_scale_(weight_scale), weight_h_(weight_h)
    {
    }

    virtual void load_lora(const T* lora_weight, const float lora_scale, IAllocator* allocator) override
    {
        // std::cout << "load_lora fp8" << std::endl;
        size_t lora_size = get_weight_size();
        T* tmp_buffer = (T*)allocator->reMallocWithName("global_shared_weight_buffer_", sizeof(T) * lora_size, false);
        cudaMemcpyAsync(tmp_buffer, weight_h_, sizeof(T) * lora_size, cudaMemcpyHostToDevice, cudaStreamDefault);
        invokeLoadLora<T>(tmp_buffer, lora_weight, lora_size, lora_scale, cudaStreamDefault);
        cudaMemsetAsync(weight_scale_, 0, sizeof(float), cudaStreamDefault);
        invokeGetFP8WeightScale(weight_scale_, tmp_buffer, lora_size, cudaStreamDefault);
        invokeCudaD2DScaleCpyConvert(weight_, tmp_buffer, weight_scale_, true, lora_size, cudaStreamDefault);
        loaded_lora = true;
    }

    virtual void clear_lora(IAllocator* allocator) override
    {
        if (!loaded_lora) {
            return;
        }

        size_t lora_size = get_weight_size();
        T* tmp_buffer = (T*)allocator->reMallocWithName("global_shared_weight_buffer_", sizeof(T) * lora_size, false);
        cudaMemcpyAsync(tmp_buffer, weight_h_, sizeof(T) * lora_size, cudaMemcpyHostToDevice, cudaStreamDefault);
        cudaMemsetAsync(weight_scale_, 0, sizeof(float), cudaStreamDefault);
        invokeGetFP8WeightScale(weight_scale_, tmp_buffer, lora_size, cudaStreamDefault);
        invokeCudaD2DScaleCpyConvert(weight_, tmp_buffer, weight_scale_, true, lora_size, cudaStreamDefault);
        loaded_lora = false;
    }
};

template class LoraWeight<float>;
template class LoraWeight<half>;
template class LoraWeight<__nv_bfloat16>;

template class FP8LoraWeight<float>;
template class FP8LoraWeight<half>;
template class FP8LoraWeight<__nv_bfloat16>;

template<typename T>
class IFLoraWeight: public IFBaseWeight<T> {
protected:
    std::unordered_map<std::string, IFLoraWeight*>  lora_layer_map;
    std::unordered_map<std::string, LoraWeight<T>*> lora_weight_map;
    IAllocator*                                     allocator_;

public:
    virtual void loadLoraByName(const std::vector<std::string>& lora_layer_names,
                                const int                       lora_layer_idx,
                                const T*                        lora_weight,
                                const float&                    lora_scale)
    {
        std::string cur_name = lora_layer_names[lora_layer_idx];
        // loaded_controlnet_weights.find(cur_weight_name) == loaded_controlnet_weights.end()
        if (lora_layer_map.find(cur_name) != lora_layer_map.end()) {
            lora_layer_map[cur_name]->loadLoraByName(lora_layer_names, lora_layer_idx + 1, lora_weight, lora_scale);
        }
        else if (lora_weight_map.find(cur_name) != lora_weight_map.end()) {
            lora_weight_map[cur_name]->load_lora(lora_weight, lora_scale, allocator_);
        }
        else {
            std::cout << "lora name cannot find weights to load: " << std::endl;
            for (int i = 0; i < lora_layer_names.size(); i++) {
                std::cout << lora_layer_names[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "cur layer idx " << lora_layer_idx << " and cur layer name: " << cur_name << std::endl;
        }
    }

    virtual void clear_lora()
    {
        for (auto it = lora_layer_map.begin(); it != lora_layer_map.end(); it++) {
            it->second->clear_lora();
        }
        for (auto it = lora_weight_map.begin(); it != lora_weight_map.end(); it++) {
            it->second->clear_lora(allocator_);
        }
    }
};
}  // namespace lyradiff