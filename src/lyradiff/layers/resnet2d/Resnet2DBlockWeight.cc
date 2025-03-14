#include "Resnet2DBlockWeight.h"
#include "src/lyradiff/kernels/lora/load_lora.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"

using namespace std;

namespace lyradiff {

template<typename T>
Resnet2DBlockWeight<T>::Resnet2DBlockWeight(const size_t in_channels,
                                            const size_t out_channels,
                                            bool         has_temb,
                                            IAllocator*  allocator):
    in_channels_(in_channels), out_channels_(out_channels), conv_shortcut_(false), has_temb_(has_temb)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    this->allocator_ = allocator;
    // 目前我先把参数写死，后续若有变动，也可以重载构造函数来实现
    conv1_kernel_h_ = 3;
    conv1_kernel_w_ = 3;

    conv2_kernel_h_ = 3;
    conv2_kernel_w_ = 3;

    time_emb_in_dim_ = 1280;

    if (in_channels != out_channels) {
        conv_shortcut_          = true;
        conv_shortcut_kernel_h_ = 1;
        conv_shortcut_kernel_w_ = 1;
    }

    conv1_size         = out_channels_ * conv1_kernel_h_ * conv1_kernel_w_ * in_channels_;
    conv2_size         = out_channels_ * conv2_kernel_h_ * conv2_kernel_w_ * out_channels_;
    time_emb_proj_size = out_channels_ * time_emb_in_dim_;
    conv_shortcut_size = out_channels_ * conv_shortcut_kernel_h_ * conv_shortcut_kernel_w_ * in_channels_;
}

template<typename T>
Resnet2DBlockWeight<T>::~Resnet2DBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {

        deviceFree(gnorm1_gamma);
        deviceFree(gnorm1_beta);

        deviceFree(gnorm2_gamma);
        deviceFree(gnorm2_beta);

        deviceFree(conv1_weight);
        deviceFree(conv1_bias);

        deviceFree(conv2_weight);
        deviceFree(conv2_bias);

        if (conv_shortcut_) {
            deviceFree(conv_shortcut_weight);
            deviceFree(conv_shortcut_bias);
            conv_shortcut_weight = nullptr;
            conv_shortcut_bias   = nullptr;
        }
        if (has_temb_) {
            deviceFree(time_emb_proj_weight);
            deviceFree(time_emb_proj_bias);
            time_emb_proj_weight = nullptr;
            time_emb_proj_bias   = nullptr;
        }

        gnorm1_gamma = nullptr;
        gnorm1_beta  = nullptr;

        gnorm2_gamma = nullptr;
        gnorm2_beta  = nullptr;

        conv1_weight = nullptr;
        conv1_bias   = nullptr;

        conv2_weight = nullptr;
        conv2_bias   = nullptr;
    }

    if (is_maintain_lora) {
        deviceFree(conv1_lora_buf_);
        deviceFree(conv2_lora_buf_);

        if (conv_shortcut_)
            deviceFree(conv_shortcut_lora_buf_);
        if (has_temb_)
            deviceFree(time_emb_proj_lora_buf_);

        is_maintain_lora = false;
    }
}

template<typename T>
void Resnet2DBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    deviceMalloc(&gnorm1_gamma, in_channels_);
    deviceMalloc(&gnorm1_beta, in_channels_);

    deviceMalloc(&gnorm2_gamma, out_channels_);
    deviceMalloc(&gnorm2_beta, out_channels_);

    deviceMalloc(&conv1_weight, out_channels_ * conv1_kernel_h_ * conv1_kernel_w_ * in_channels_);
    deviceMalloc(&conv1_bias, out_channels_);

    deviceMalloc(&conv2_weight, out_channels_ * conv2_kernel_h_ * conv2_kernel_w_ * out_channels_);
    deviceMalloc(&conv2_bias, out_channels_);

    if (conv_shortcut_) {
        deviceMalloc(&conv_shortcut_weight,
                     out_channels_ * conv_shortcut_kernel_h_ * conv_shortcut_kernel_w_ * in_channels_);
        deviceMalloc(&conv_shortcut_bias, out_channels_);
    }
    if (has_temb_) {
        deviceMalloc(&time_emb_proj_weight, out_channels_ * time_emb_in_dim_);
        deviceMalloc(&time_emb_proj_bias, out_channels_);
    }

    // this->lora_weight_size_map = {{"conv1", conv1_size},
    //                               {"conv2", conv2_size},
    //                               {"conv_shortcut", conv_shortcut_size},
    //                               {"time_emb_proj", time_emb_proj_size}};
    // this->lora_weight_map      = {{"conv1", conv1_weight},
    //                               {"conv2", conv2_weight},
    //                               {"conv_shortcut", conv_shortcut_weight},
    //                               {"time_emb_proj", time_emb_proj_weight}};

    this->lora_weight_map = {
        {"conv1", new LoraWeight<T>({out_channels_, conv1_kernel_h_, conv1_kernel_w_, in_channels_}, conv1_weight)},
        {"conv2", new LoraWeight<T>({out_channels_, conv2_kernel_h_, conv2_kernel_w_, out_channels_}, conv2_weight)},
        {"conv_shortcut",
         new LoraWeight<T>({out_channels_, conv_shortcut_kernel_h_, conv_shortcut_kernel_w_, in_channels_},
                           conv_shortcut_weight)},
        {"time_emb_proj", new LoraWeight<T>({out_channels_, time_emb_in_dim_}, time_emb_proj_weight)}};

    is_maintain_buffer = true;
}

template<typename T>
void Resnet2DBlockWeight<T>::loadWeights(const std::string& prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // norm weights
    loadWeightFromBin<T>(gnorm1_gamma, {in_channels_}, prefix + "norm1.weight.bin", model_file_type);
    loadWeightFromBin<T>(gnorm1_beta, {in_channels_}, prefix + "norm1.bias.bin", model_file_type);
    loadWeightFromBin<T>(gnorm2_gamma, {out_channels_}, prefix + "norm2.weight.bin", model_file_type);
    loadWeightFromBin<T>(gnorm2_beta, {out_channels_}, prefix + "norm2.bias.bin", model_file_type);

    // conv weights, for now NHWC
    loadWeightFromBin<T>(conv1_weight,
                         {out_channels_ * conv1_kernel_h_ * conv1_kernel_w_ * in_channels_},
                         prefix + "conv1.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv1_bias, {out_channels_}, prefix + "conv1.bias.bin", model_file_type);

    loadWeightFromBin<T>(conv2_weight,
                         {out_channels_ * conv2_kernel_h_ * conv2_kernel_w_ * out_channels_},
                         prefix + "conv2.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv2_bias, {out_channels_}, prefix + "conv2.bias.bin", model_file_type);

    if (conv_shortcut_) {
        loadWeightFromBin<T>(conv_shortcut_weight,
                             {out_channels_ * conv_shortcut_kernel_h_ * conv_shortcut_kernel_w_ * in_channels_},
                             prefix + "conv_shortcut.weight.bin",
                             model_file_type);
        loadWeightFromBin<T>(conv_shortcut_bias, {out_channels_}, prefix + "conv_shortcut.bias.bin", model_file_type);
    }
    if (has_temb_) {
        loadWeightFromBin<T>(time_emb_proj_weight,
                             {out_channels_ * time_emb_in_dim_},
                             prefix + "time_emb_proj.weight.bin",
                             model_file_type);

        loadWeightFromBin<T>(time_emb_proj_bias, {out_channels_}, prefix + "time_emb_proj.bias.bin", model_file_type);
    }
}

template<typename T>
void Resnet2DBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                  std::unordered_map<std::string, void*>& weights,
                                                  cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_gnorm1_gamma = weights[prefix + "norm1.weight"];
    void* tmp_gnorm1_beta  = weights[prefix + "norm1.bias"];
    void* tmp_gnorm2_gamma = weights[prefix + "norm2.weight"];
    void* tmp_gnorm2_beta  = weights[prefix + "norm2.bias"];
    void* tmp_conv1_weight = weights[prefix + "conv1.weight"];
    void* tmp_conv1_bias   = weights[prefix + "conv1.bias"];
    void* tmp_conv2_weight = weights[prefix + "conv2.weight"];
    void* tmp_conv2_bias   = weights[prefix + "conv2.bias"];

    weight_loader_manager_glob->doCudaMemcpy(gnorm1_gamma, tmp_gnorm1_gamma, sizeof(T) * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(gnorm1_beta, tmp_gnorm1_beta, sizeof(T) * in_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(gnorm2_gamma, tmp_gnorm2_gamma, sizeof(T) * out_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(gnorm2_beta, tmp_gnorm2_beta, sizeof(T) * out_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv1_weight,
               tmp_conv1_weight,
               sizeof(T) * out_channels_ * conv1_kernel_h_ * conv1_kernel_w_ * in_channels_,
               memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv1_bias, tmp_conv1_bias, sizeof(T) * out_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv2_weight,
               tmp_conv2_weight,
               sizeof(T) * out_channels_ * conv2_kernel_h_ * conv2_kernel_w_ * out_channels_,
               memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv2_bias, tmp_conv2_bias, sizeof(T) * out_channels_, memcpyKind);

    if (conv_shortcut_) {
        void* tmp_conv_shortcut_weight = weights[prefix + "conv_shortcut.weight"];
        void* tmp_conv_shortcut_bias   = weights[prefix + "conv_shortcut.bias"];
        weight_loader_manager_glob->doCudaMemcpy(conv_shortcut_weight,
                   tmp_conv_shortcut_weight,
                   sizeof(T) * out_channels_ * conv_shortcut_kernel_h_ * conv_shortcut_kernel_w_ * in_channels_,
                   memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(conv_shortcut_bias, tmp_conv_shortcut_bias, sizeof(T) * out_channels_, memcpyKind);
    }
    if (has_temb_) {
        void* tmp_time_emb_proj_weight = weights[prefix + "time_emb_proj.weight"];
        void* tmp_time_emb_proj_bias   = weights[prefix + "time_emb_proj.bias"];

        weight_loader_manager_glob->doCudaMemcpy(
            time_emb_proj_weight, tmp_time_emb_proj_weight, sizeof(T) * out_channels_ * time_emb_in_dim_, memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(time_emb_proj_bias, tmp_time_emb_proj_bias, sizeof(T) * out_channels_, memcpyKind);
    }
}

template<typename T>
void Resnet2DBlockWeight<T>::mallocLoraBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_lora) {
        deviceMalloc(&conv1_lora_buf_, out_channels_ * conv1_kernel_h_ * conv1_kernel_w_ * in_channels_);
        deviceMalloc(&conv2_lora_buf_, out_channels_ * conv2_kernel_h_ * conv2_kernel_w_ * out_channels_);
        if (has_temb_) {
            deviceMalloc(&time_emb_proj_lora_buf_, out_channels_ * time_emb_in_dim_);
        }
        if (conv_shortcut_) {
            deviceMalloc(&conv_shortcut_lora_buf_,
                         out_channels_ * conv_shortcut_kernel_h_ * conv_shortcut_kernel_w_ * in_channels_);
        }
        is_maintain_lora = true;
    }
}

template<typename T>
void Resnet2DBlockWeight<T>::loadLoraFromWeight(std::string                          lora_path,
                                                std::string                          prefix,
                                                std::unordered_map<std::string, T*>& lora_weights,
                                                float                                lora_alpha,
                                                FtCudaDataType                       lora_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!checkIfFileExist(lora_path + prefix + "conv1.bin")) {
        return;
    }

    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }

    loadWeightFromBin<T>(conv1_lora_buf_, {conv1_size}, lora_path + prefix + "conv1.bin", lora_file_type);
    loadWeightFromBin<T>(conv2_lora_buf_, {conv2_size}, lora_path + prefix + "conv2.bin", lora_file_type);

    T* conv_1 = (T*)malloc(sizeof(T) * conv1_size);
    T* conv_2 = (T*)malloc(sizeof(T) * conv2_size);

    lora_weights[prefix + "conv1"] = conv_1;
    lora_weights[prefix + "conv2"] = conv_2;

    // 再缓存到本地
    cudaMemcpy(conv_1, conv1_lora_buf_, sizeof(T) * conv1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(conv_2, conv2_lora_buf_, sizeof(T) * conv2_size, cudaMemcpyDeviceToHost);

    invokeLoadLora<T>(conv1_weight, conv1_lora_buf_, conv1_size, lora_alpha);
    invokeLoadLora<T>(conv2_weight, conv2_lora_buf_, conv2_size, lora_alpha);

    if (has_temb_) {
        loadWeightFromBin<T>(
            time_emb_proj_lora_buf_, {time_emb_proj_size}, lora_path + prefix + "time_emb_proj.bin", lora_file_type);

        T* time_emb_proj = (T*)malloc(sizeof(T) * time_emb_proj_size);
        cudaMemcpy(time_emb_proj, time_emb_proj_lora_buf_, sizeof(T) * time_emb_proj_size, cudaMemcpyDeviceToHost);
        lora_weights[prefix + "time_emb_proj"] = time_emb_proj;
        invokeLoadLora<T>(time_emb_proj_weight, time_emb_proj_lora_buf_, time_emb_proj_size, lora_alpha);
    }

    if (conv_shortcut_) {
        loadWeightFromBin<T>(
            conv_shortcut_lora_buf_, {conv_shortcut_size}, lora_path + prefix + "conv_shortcut.bin", lora_file_type);

        T* conv_shortcut = (T*)malloc(sizeof(T) * conv_shortcut_size);
        cudaMemcpy(conv_shortcut, conv_shortcut_lora_buf_, sizeof(T) * conv_shortcut_size, cudaMemcpyDeviceToHost);
        lora_weights[prefix + "conv_shortcut"] = conv_shortcut;
        invokeLoadLora<T>(conv_shortcut_weight, conv_shortcut_lora_buf_, conv_shortcut_size, lora_alpha);
    }
}

template<typename T>
void Resnet2DBlockWeight<T>::loadLoraFromCache(std::string                          prefix,
                                               std::unordered_map<std::string, T*>& lora_weights,
                                               float                                lora_alpha,
                                               bool                                 from_outside)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (lora_weights.find(prefix + "conv1") == lora_weights.end()) {
        return;
    }

    if (!is_maintain_lora) {
        mallocLoraBuffer();
    }

    T* conv1 = lora_weights[prefix + "conv1"];
    T* conv2 = lora_weights[prefix + "conv2"];
    cudaMemcpy(conv1_lora_buf_, conv1, sizeof(T) * conv1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_lora_buf_, conv2, sizeof(T) * conv2_size, cudaMemcpyHostToDevice);
    invokeLoadLora<T>(conv1_weight, conv1_lora_buf_, conv1_size, lora_alpha);
    invokeLoadLora<T>(conv2_weight, conv2_lora_buf_, conv2_size, lora_alpha);

    if (conv_shortcut_) {
        T* conv_shortcut = lora_weights[prefix + "conv_shortcut"];
        cudaMemcpy(conv_shortcut_lora_buf_, conv_shortcut, sizeof(T) * conv_shortcut_size, cudaMemcpyHostToDevice);
        invokeLoadLora<T>(conv_shortcut_weight, conv_shortcut_lora_buf_, conv_shortcut_size, lora_alpha);
    }

    if (has_temb_) {
        T* time_emb_proj = lora_weights[prefix + "time_emb_proj"];
        cudaMemcpy(time_emb_proj_lora_buf_, time_emb_proj, sizeof(T) * time_emb_proj_size, cudaMemcpyHostToDevice);
        invokeLoadLora<T>(time_emb_proj_weight, time_emb_proj_lora_buf_, time_emb_proj_size, lora_alpha);
    }
}

template class Resnet2DBlockWeight<float>;
template class Resnet2DBlockWeight<half>;
}  // namespace lyradiff