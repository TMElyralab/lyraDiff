#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"
namespace lyradiff {

template<typename T>
class Resnet2DBlockWeight: public IFLoraWeight<T> {
private:
    size_t in_channels_;
    size_t out_channels_;
    bool   conv_shortcut_;

    size_t conv1_kernel_h_;
    size_t conv1_kernel_w_;

    size_t conv2_kernel_h_;
    size_t conv2_kernel_w_;

    size_t conv_shortcut_kernel_h_;
    size_t conv_shortcut_kernel_w_;

    size_t time_emb_in_dim_;

    bool has_temb_;

    size_t conv1_size;
    size_t conv2_size;
    size_t time_emb_proj_size;
    size_t conv_shortcut_size;

protected:
    bool is_maintain_buffer = false;
    bool is_maintain_lora   = false;

public:
    T* gnorm1_gamma         = nullptr;
    T* gnorm1_beta          = nullptr;
    T* gnorm2_gamma         = nullptr;
    T* gnorm2_beta          = nullptr;
    T* conv1_weight         = nullptr;
    T* conv1_bias           = nullptr;
    T* conv2_weight         = nullptr;
    T* conv2_bias           = nullptr;
    T* conv_shortcut_weight = nullptr;
    T* conv_shortcut_bias   = nullptr;
    T* time_emb_proj_weight = nullptr;
    T* time_emb_proj_bias   = nullptr;

    T* conv1_lora_buf_         = nullptr;
    T* conv2_lora_buf_         = nullptr;
    T* conv_shortcut_lora_buf_ = nullptr;
    T* time_emb_proj_lora_buf_ = nullptr;

    Resnet2DBlockWeight() = default;
    Resnet2DBlockWeight(const size_t in_channels,
                        const size_t out_channels,
                        bool         has_temb  = true,
                        IAllocator*  allocator = nullptr);

    ~Resnet2DBlockWeight();

    virtual void loadWeights(const std::string& prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
    virtual void mallocLoraBuffer();
    virtual void loadLoraFromWeight(std::string                          lora_path,
                                    std::string                          prefix,
                                    std::unordered_map<std::string, T*>& lora_weights,
                                    float                                lora_alpha,
                                    FtCudaDataType                       lora_file_type);
    virtual void loadLoraFromCache(std::string                          prefix,
                                   std::unordered_map<std::string, T*>& lora_weights,
                                   float                                lora_alpha,
                                   bool                                 from_outside = true);
};

}  // namespace lyradiff