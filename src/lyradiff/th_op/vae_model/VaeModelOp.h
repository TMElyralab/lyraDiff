#pragma once

#include "src/lyradiff/models/unet_2d_conditional_model/Unet2dConditionalModel.h"
#include "src/lyradiff/models/vae_model/VaeModel.h"
#include "src/lyradiff/th_op/th_utils.h"
// #include "src/lyradiff/utils/auth_utils.h"
#include "src/lyradiff/utils/cublasMMWrapper.h"
#include <cstdlib>
#include <nvml.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace th = torch;
using std::vector;
using namespace lyradiff;
using namespace std;

namespace torch_ext {

// *************** FOR ERROR CHECKING *******************
#ifndef NVML_RT_CALL
#define NVML_RT_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>(call);                                                                 \
        if (status != NVML_SUCCESS)                                                                                    \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                       \
                    "with "                                                                                            \
                    "%s (%d).\n",                                                                                      \
                    #call,                                                                                             \
                    __LINE__,                                                                                          \
                    __FILE__,                                                                                          \
                    nvmlErrorString(status),                                                                           \
                    status);                                                                                           \
    }
#endif  // NVML_RT_CALL
// *************** FOR ERROR CHECKING *******************

class IFVae {
public:
    virtual ~IFVae() {}

    virtual th::Tensor forward(th::Tensor hidden_states) = 0;

    virtual void reload_vae_model(const std::string model_path, const std::string model_dtype)   = 0;
    virtual void reload_vae_encoder(const std::string model_path, const std::string model_dtype) = 0;
    virtual void reload_vae_decoder(const std::string model_path, const std::string model_dtype) = 0;

    virtual void reload_vae_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)   = 0;
    virtual void reload_vae_encoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device) = 0;
    virtual void reload_vae_decoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device) = 0;

    virtual void load_s3diff_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights_alpha,
                                                  th::Dict<std::string, th::Tensor> weights_beta) = 0;

    virtual th::Tensor vae_decode(th::Tensor                                      hidden_states,
                                  th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                                  th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt) = 0;

    virtual th::Tensor vae_encode(th::Tensor                                      hidden_states,
                                  th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                                  th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt) = 0;
};

template<typename T>
class LyraDiffVaeModel: public IFVae {

private:
    cudaStream_t stream_ = cudaStreamDefault;

    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudnnHandle_t    cudnn_handle_;

    cublasAlgoMap*                  cublas_algo_map_;
    Allocator<AllocatorType::CUDA>* allocator_;
    std::mutex*                     cublas_wrapper_mutex_;
    lyradiff::cublasMMWrapper*      cublas_wrapper_;
    std::mutex*                     lock_;

    VaeModel<T>*       vae_model_;
    VaeModelWeight<T>* vae_model_weight_;

    std::string machine_id;
    std::string token_;

    bool is_upcast_ = false;

    int retry_limit = 5;  // 暂时auth 给5次使用机会

public:
    LyraDiffVaeModel(const std::string inference_dtype, const bool is_upcast)
    {
        cudnnCreate(&cudnn_handle_);
        cublasCreate(&cublas_handle_);
        cublasLtCreate(&cublaslt_handle_);

        cublas_algo_map_ = new cublasAlgoMap(GEMM_CONFIG);
        // allocator_       = new Allocator<AllocatorType::CUDA>(getDevice());
        if (allocator_glob == nullptr)
            allocator_glob = new Allocator<AllocatorType::CUDA>(getDevice());
        allocator_ = allocator_glob;

        cublas_wrapper_mutex_ = new std::mutex();
        if (cublas_wrapper_glob == nullptr) {
            cublas_wrapper_glob = new lyradiff::cublasMMWrapper(
                cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
        }
        cublas_wrapper_ = cublas_wrapper_glob;

        is_upcast_ = is_upcast;

        lock_ = new std::mutex();

        vae_model_ = new VaeModel<T>(cudnn_handle_, stream_, cublas_wrapper_, allocator_, false, false, is_upcast);
        vae_model_weight_ = new VaeModelWeight<T>(is_upcast);
    }

    ~LyraDiffVaeModel() override
    {
        delete vae_model_;
        delete vae_model_weight_;

        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
        delete cublas_wrapper_;
        // delete allocator_;
        delete lock_;

        cublasLtDestroy(cublaslt_handle_);
        cublasDestroy(cublas_handle_);
        cudnnDestroy(cudnn_handle_);
    }

    th::Tensor forward(th::Tensor hidden_states)
    {
        return torch::empty({1, 1});
    }

    void reload_vae_model(const std::string model_path, const std::string model_dtype)
    {
        lock_->lock();

        lyradiff::FtCudaDataType model_file_type = lyradiff::FtCudaDataType::FP32;

        if (model_dtype == "fp32") {
            model_file_type = lyradiff::FtCudaDataType::FP32;
        }
        else if (model_dtype == "fp16") {
            model_file_type = lyradiff::FtCudaDataType::FP16;
        }
        else {
            throw "wrong model_dtype";
        }

        vae_model_weight_->loadWeights(model_path, model_file_type);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void reload_vae_encoder(const std::string model_path, const std::string model_dtype)
    {
        lock_->lock();

        lyradiff::FtCudaDataType model_file_type = lyradiff::FtCudaDataType::FP32;

        if (model_dtype == "fp32") {
            model_file_type = lyradiff::FtCudaDataType::FP32;
        }
        else if (model_dtype == "fp16") {
            model_file_type = lyradiff::FtCudaDataType::FP16;
        }
        else {
            throw "wrong model_dtype";
        }

        vae_model_weight_->loadEncoderWeights(model_path, model_file_type);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void reload_vae_decoder(const std::string model_path, const std::string model_dtype)
    {
        lock_->lock();

        lyradiff::FtCudaDataType model_file_type = lyradiff::FtCudaDataType::FP32;

        if (model_dtype == "fp32") {
            model_file_type = lyradiff::FtCudaDataType::FP32;
        }
        else if (model_dtype == "fp16") {
            model_file_type = lyradiff::FtCudaDataType::FP16;
        }
        else {
            throw "wrong model_dtype";
        }

        vae_model_weight_->loadDecoderWeights(model_path, model_file_type);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void reload_vae_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
    {
        lock_->lock();

        std::unordered_map<std::string, void*> tmp_weights;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            tmp_weights[it->key()] = reinterpret_cast<void*>(it->value().data_ptr());
        }
        weight_loader_manager_glob->set_map_weights_reverse(tmp_weights);

        cudaMemcpyKind memcpyKind = cudaMemcpyHostToDevice;
        if (device.rfind("cuda", 0) == 0) {
            memcpyKind = cudaMemcpyDeviceToDevice;
        }

        vae_model_weight_->loadWeightsFromCache("", tmp_weights, memcpyKind);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void reload_vae_decoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
    {
        lock_->lock();

        std::unordered_map<std::string, void*> tmp_weights;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            tmp_weights[it->key()] = reinterpret_cast<void*>(it->value().data_ptr());
        }

        cudaMemcpyKind memcpyKind = cudaMemcpyHostToDevice;
        if (device.rfind("cuda", 0) == 0) {
            memcpyKind = cudaMemcpyDeviceToDevice;
        }

        vae_model_weight_->loadDecoderWeightsFromCache("", tmp_weights, memcpyKind);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void load_s3diff_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights_alpha,
                                          th::Dict<std::string, th::Tensor> weights_beta)
    {
        std::lock_guard<std::mutex> lock(*lock_);
        auto                        load_func = [&](th::Dict<std::string, th::Tensor>& weights, bool is_alpha) {
            std::unordered_map<std::string, T*> tmp_weights;
            for (auto it = weights.begin(); it != weights.end(); it++) {
                tmp_weights[it->key()] = reinterpret_cast<T*>(it->value().data_ptr());
            }
            vae_model_weight_->loadS3DiffLoraFromStateDict(tmp_weights, is_alpha);
        };
        load_func(weights_alpha, true);
        load_func(weights_beta, false);
    }

    void reload_vae_encoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device)
    {
        lock_->lock();

        std::unordered_map<std::string, void*> tmp_weights;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            tmp_weights[it->key()] = reinterpret_cast<void*>(it->value().data_ptr());
        }

        cudaMemcpyKind memcpyKind = cudaMemcpyHostToDevice;
        if (device.rfind("cuda", 0) == 0) {
            memcpyKind = cudaMemcpyDeviceToDevice;
        }

        vae_model_weight_->loadEncoderWeightsFromCache("", tmp_weights, memcpyKind);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    th::Tensor vae_decode(th::Tensor                                      hidden_states,
                          th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                          th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt)
    {
        size_t     batch_size = hidden_states.sizes()[0];
        size_t     height     = hidden_states.sizes()[1];
        size_t     width      = hidden_states.sizes()[2];
        th::Tensor res_hidden_state_torch =
            torch::empty({batch_size, height * 8, width * 8, 3},
                         torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
        lyradiff::Tensor output               = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor = convert_tensor<T>(hidden_states);

        TensorMap input_tensor({{"hidden_states", hidden_states_tensor}});
        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);
        context.cur_running_module = "vae";

        insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        insertParamsToMap(input_tensor, scale_params);

        weight_loader_manager_glob->ctx = &context;
        for (auto& key : context.keys()) {
            weight_loader_manager_glob->map_lora_container["vae"].set_map_de_mods(key,
                                                                                  (void*)(context.at(key).getPtr<T>()));
        }

        lock_->lock();
        vae_model_->decode(&output_tensor, &input_tensor, vae_model_weight_);
        cudaDeviceSynchronize();
        lock_->unlock();

        return res_hidden_state_torch;
    }

    th::Tensor vae_encode(th::Tensor                                      hidden_states,
                          th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                          th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt)
    {
        size_t     batch_size = hidden_states.sizes()[0];
        size_t     height     = hidden_states.sizes()[1];
        size_t     width      = hidden_states.sizes()[2];
        th::Tensor res_hidden_state_torch =
            torch::empty({batch_size, height / 8, width / 8, 8},
                         torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
        lyradiff::Tensor output               = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor = convert_tensor<T>(hidden_states);

        // cout << "vae_encode input shape " << batch_size << " " << height << " " << width << " " <<
        // hidden_states.sizes()[3] << " " << endl;

        TensorMap input_tensor({{"hidden_states", hidden_states_tensor}});
        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);
        context.cur_running_module = "vae";
        input_tensor.context_      = &context;

        insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        insertParamsToMap(input_tensor, scale_params);

        weight_loader_manager_glob->ctx = &context;
        // printf("context size: %d\n", context.keys().size());
        for (auto& key : context.keys()) {
            weight_loader_manager_glob->map_lora_container["vae"].set_map_de_mods(key,
                                                                                  (void*)(context.at(key).getPtr<T>()));
        }
        // for(auto iter: weight_loader_manager_glob->map_lora_container){
        //     printf("map_lora_container %s %d\n", iter.first.c_str(), iter.second.map_de_mods.size());
        // }

        lock_->lock();
        vae_model_->encode(&output_tensor, &input_tensor, vae_model_weight_);
        cudaDeviceSynchronize();
        lock_->unlock();

        return res_hidden_state_torch;
    }
};

class VaeModelOp: public th::jit::CustomClassHolder {
public:
    VaeModelOp(const std::string inference_dtype, const bool is_upcast);
    ~VaeModelOp();

    th::Tensor forward(th::Tensor hidden_states);  // placeholder 暂时不用

    void reload_vae_model(const std::string model_path, const std::string model_dtype);

    void reload_vae_encoder(const std::string model_path, const std::string model_dtype);

    void reload_vae_decoder(const std::string model_path, const std::string model_dtype);

    void reload_vae_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device);

    void reload_vae_encoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device);

    void reload_vae_decoder_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device);

    void load_s3diff_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights_alpha,
                                          th::Dict<std::string, th::Tensor> weights_beta);

    th::Tensor vae_decode(th::Tensor                                      hidden_states,
                          th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                          th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt);

    th::Tensor vae_encode(th::Tensor                                      hidden_states,
                          th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors = th::nullopt,
                          th::optional<th::Dict<std::string, double>>     scale_params      = th::nullopt);

private:
    // const at::ScalarType    st_;
    IFVae*      model_;
    std::string inference_dtype_;
};

}  // namespace torch_ext
