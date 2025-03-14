#pragma once

#include "src/lyradiff/models/xlunet_2d_conditional_model/XLUnet2dConditionalModel.h"
#include "src/lyradiff/th_op/th_utils.h"
#include "src/lyradiff/utils/auth_utils.h"
#include "src/lyradiff/utils/cublasMMWrapper.h"
#include <cstdlib>
#include <nvml.h>
#include <stdexcept>
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

class IFUnet {
public:
    virtual ~IFUnet() {}
    virtual th::Tensor forward(th::Tensor                                      hidden_states,
                               th::Tensor                                      encoder_hidden_states,
                               const double                                    timestep,
                               th::Tensor&                                     aug_emb,
                               th::optional<vector<std::string>>               controlnet_names,
                               th::optional<vector<th::Tensor>>                controlnet_conds,
                               th::optional<vector<th::Tensor>>                controlnet_augs,
                               th::optional<vector<vector<double>>>            controlnet_scales,
                               th::optional<bool>                              controlnet_guess_mode,
                               th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                               th::optional<th::Dict<std::string, double>>     scale_params) = 0;

    virtual std::vector<th::Tensor> controlnet_forward(th::Tensor          hidden_states,
                                                       th::Tensor          encoder_hidden_states,
                                                       th::Tensor          conditioning_img,
                                                       th::Tensor          controlnet_aug,
                                                       const double        timestep,
                                                       std::vector<double> controlnet_scales,
                                                       std::string         controlnet_name) = 0;

    virtual th::Tensor unet_forward(th::Tensor                                      hidden_states,
                                    th::Tensor                                      encoder_hidden_states,
                                    th::Tensor                                      aug_emb,
                                    const double                                    timestep,
                                    th::optional<vector<th::Tensor>>                controlnet_results,
                                    th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                                    th::optional<th::Dict<std::string, double>>     scale_params) = 0;

    virtual void load_lora(const std::string lora_path,
                           const std::string lora_name,
                           const double      lora_strength,
                           const std::string model_dtype) = 0;

    virtual void load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights, const double lora_strength) = 0;

    virtual void
    load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength) = 0;

    virtual void unload_lora(const std::string lora_name, const bool clean_mem_cache) = 0;

    virtual void load_ip_adapter(const std::string& ip_adapter_path,
                                 const std::string& ip_adapter_name,
                                 const double&      scale,
                                 const std::string& model_dtype) = 0;

    virtual void unload_ip_adapter(const std::string& ip_adapter_name) = 0;

    virtual void reload_unet_model(const std::string model_path, const std::string model_dtype) = 0;

    virtual void reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device) = 0;

    virtual void load_controlnet_model(const std::string controlnet_name,
                                       const std::string model_path,
                                       const std::string model_dtype,
                                       const std::string controlnet_mode = "large") = 0;

    virtual void unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache) = 0;

    virtual void load_controlnet_model_from_state_dict(const std::string                 controlnet_name,
                                                       th::Dict<std::string, th::Tensor> weights,
                                                       const std::string                 device,
                                                       const std::string                 controlnet_mode = "large") = 0;

    virtual void clean_lora_cache() = 0;

    virtual void clean_controlnet_cache() = 0;

    virtual std::vector<std::string> get_loaded_controlnet() = 0;

    virtual std::vector<std::string> get_loaded_lora() = 0;
};

template<typename T>
class LyraDiffXLUnet2dConditionalModel: public IFUnet {

private:
    cudaStream_t stream_ = cudaStreamDefault;
    // cudaStream_t stream_assistant_;

    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    cudnnHandle_t    cudnn_handle_;

    cublasAlgoMap*                  cublas_algo_map_;
    Allocator<AllocatorType::CUDA>* allocator_;
    std::mutex*                     cublas_wrapper_mutex_;
    lyradiff::cublasMMWrapper*      cublas_wrapper_;
    std::mutex*                     lock_;

    XLUnet2dConditionalModel<T>*       model_;
    XLUnet2dConditionalModelWeight<T>* model_weight_;

    std::unordered_map<std::string, XLControlNetModelWeight<T>*> loaded_controlnet_weights;

    std::vector<XLControlNetModelWeight<T>*> cache_controlnet_weights;

    std::unordered_map<std::string, std::unordered_map<std::string, T*>> loaded_lora;
    std::unordered_map<std::string, float>                               loaded_lora_strength;

    int64_t max_controlnet_count_ = 3;

    std::string machine_id;
    std::string token_;

    bool loop_ = true;

    size_t input_channels_  = 4;
    size_t output_channels_ = 4;

    int64_t quant_level_ = 0;

    LyraQuantType quant_level_enum_ = LyraQuantType::NONE;

    bool is_internal_ = false;

public:
    LyraDiffXLUnet2dConditionalModel(const std::string inference_dtype,
                                     const int64_t     input_channels,
                                     const int64_t     output_channels,
                                     const int64_t     quant_level)
    {
        // max_controlnet_count_ = max_controlnet_count;
        quant_level_ = quant_level;
        cudnnCreate(&cudnn_handle_);
        cublasCreate(&cublas_handle_);
        cublasLtCreate(&cublaslt_handle_);

        cublas_algo_map_ = new cublasAlgoMap(GEMM_CONFIG);
        allocator_       = new Allocator<AllocatorType::CUDA>(getDevice());

        cublas_wrapper_mutex_ = new std::mutex();
        cublas_wrapper_       = new lyradiff::cublasMMWrapper(
            cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);

        lock_ = new std::mutex();

        input_channels_  = input_channels;
        output_channels_ = output_channels;

        quant_level_enum_ = LyraQuantType::NONE;
        if (quant_level == 1) {
            quant_level_enum_ = LyraQuantType::INT8_SMOOTH_QUANT_LEVEL1;
        }
        else if (quant_level == 2) {
            quant_level_enum_ = LyraQuantType::INT8_SMOOTH_QUANT_LEVEL2;
        }
        else if (quant_level == 3) {
            quant_level_enum_ = LyraQuantType::INT8_SMOOTH_QUANT_LEVEL3;
        }

        model_ = new XLUnet2dConditionalModel<T>(cudnn_handle_,
                                                 stream_,
                                                 cublas_wrapper_,
                                                 allocator_,
                                                 false,
                                                 false,
                                                 false,
                                                 input_channels,
                                                 output_channels,
                                                 quant_level_enum_);

        model_weight_ = new XLUnet2dConditionalModelWeight<T>(
            input_channels, output_channels, false, quant_level_enum_, allocator_);
    }

    ~LyraDiffXLUnet2dConditionalModel() override
    {
        delete model_;
        delete model_weight_;

        loaded_lora_strength.clear();
        clean_lora_cache();
        clean_controlnet_cache();

        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
        delete cublas_wrapper_;
        delete allocator_;
        delete lock_;

        cublasLtDestroy(cublaslt_handle_);
        cublasDestroy(cublas_handle_);
        cudnnDestroy(cudnn_handle_);
    }

    th::Tensor forward(th::Tensor                                      hidden_states,
                       th::Tensor                                      encoder_hidden_states,
                       const double                                    timestep,
                       th::Tensor&                                     aug_emb,
                       th::optional<vector<std::string>>               controlnet_names,
                       th::optional<vector<th::Tensor>>                controlnet_conds,
                       th::optional<vector<th::Tensor>>                controlnet_augs,
                       th::optional<vector<vector<double>>>            controlnet_scales,
                       th::optional<bool>                              controlnet_guess_mode,
                       th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                       th::optional<th::Dict<std::string, double>>     scale_params)
    {
        // cudaStreamSynchronize(stream_);
        std::vector<Tensor>                      input_conds;
        std::vector<Tensor>                      input_controlnet_augs;
        std::vector<std::vector<float>>          input_controlnet_scales;
        std::vector<XLControlNetModelWeight<T>*> input_controlnet_weights;
        bool                                     guess_mode = false;
        if (controlnet_guess_mode.has_value()) {
            guess_mode = controlnet_guess_mode.value();
        }
        if (controlnet_names.has_value() && controlnet_conds.has_value() && controlnet_scales.has_value()
            && controlnet_augs.has_value()) {
            if (controlnet_names.value().size() != controlnet_conds.value().size()) {
                throw std::invalid_argument("controlnet_names and controlnet_conds has different size");
            }

            if (controlnet_names.value().size() != controlnet_conds.value().size()) {
                throw std::invalid_argument("controlnet_names and controlnet_conds has different size");
            }

            if (controlnet_names.value().size() != controlnet_scales.value().size()) {
                throw std::invalid_argument("controlnet_names and controlnet_scales has different size");
            }

            if (controlnet_names.value().size() != controlnet_augs.value().size()) {
                throw std::invalid_argument("controlnet_names and controlnet_augs has different size");
            }

            int len = controlnet_names.value().size();

            for (int i = 0; i < len; i++) {
                std::string cur_weight_name = controlnet_names.value()[i];
                if (loaded_controlnet_weights.find(cur_weight_name) == loaded_controlnet_weights.end()) {
                    cout << cur_weight_name << " controlnet weight is not loaded";
                    throw "some input weights are not loaded";
                }

                input_controlnet_weights.push_back(loaded_controlnet_weights[cur_weight_name]);
                std::vector<float> cur_scales;
                for (int k = 0; k < controlnet_scales.value()[i].size(); k++) {
                    cur_scales.push_back(controlnet_scales.value()[i][k]);
                }
                input_controlnet_scales.push_back(cur_scales);

                th::Tensor cur_input_cond     = controlnet_conds.value()[i];
                th::Tensor cur_controlnet_aug = controlnet_augs.value()[i];

                input_conds.push_back(convert_tensor<T>(cur_input_cond));
                input_controlnet_augs.push_back(convert_tensor<T>(cur_controlnet_aug));
            }
        }

        size_t batch_size = hidden_states.size(0);
        size_t height     = hidden_states.size(1);
        size_t width      = hidden_states.size(2);

        // cudaStreamSynchronize(stream_);
        th::Tensor res_hidden_state_torch =
            torch::empty({batch_size, height, width, output_channels_},
                         torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
        lyradiff::Tensor output                       = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor         = convert_tensor<T>(hidden_states);
        lyradiff::Tensor encoder_hidden_states_tensor = convert_tensor<T>(encoder_hidden_states);
        lyradiff::Tensor aug_emb_tensor               = convert_tensor<T>(aug_emb);

        TensorMap input_tensor(
            {{"hidden_states", hidden_states_tensor}, {"encoder_hidden_states", encoder_hidden_states_tensor}});
        TensorMap add_tensor({{"aug_emb", aug_emb_tensor}});
        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);

        insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        insertParamsToMap(input_tensor, scale_params);

        lock_->lock();
        model_->forward(&output_tensor,
                        &input_tensor,
                        timestep,
                        &add_tensor,
                        model_weight_,
                        &input_conds,
                        &input_controlnet_augs,
                        &input_controlnet_scales,
                        &input_controlnet_weights,
                        guess_mode);

        cudaDeviceSynchronize();
        lock_->unlock();

        return res_hidden_state_torch;
    }

    std::vector<th::Tensor> controlnet_forward(th::Tensor          hidden_states,
                                               th::Tensor          encoder_hidden_states,
                                               th::Tensor          conditioning_img,
                                               th::Tensor          controlnet_aug,
                                               const double        timestep,
                                               std::vector<double> controlnet_scales,
                                               std::string         controlnet_name) override
    {
        if (loaded_controlnet_weights.find(controlnet_name) == loaded_controlnet_weights.end()) {
            cout << controlnet_name << " controlnet weight is not loaded";
            throw "some input weights are not loaded";
        }

        size_t batch_size = hidden_states.size(0);
        size_t height     = hidden_states.size(1);
        size_t width      = hidden_states.size(2);

        std::vector<std::vector<int64_t>> controlnet_output_sizes =
            model_->get_controlnet_results_shape(batch_size, height, width);

        std::vector<float> input_controlnet_scales;
        for (int k = 0; k < controlnet_scales.size(); k++) {
            input_controlnet_scales.push_back(controlnet_scales[k]);
        }

        LYRA_CHECK_WITH_INFO(input_controlnet_scales.size() == controlnet_output_sizes.size(),
                             "controlnet_scales len not same as controlnet_output_sizes");

        std::vector<lyradiff::Tensor> output_tensors;
        std::vector<th::Tensor>       output_th_tensors;

        for (int i = 0; i < controlnet_output_sizes.size(); i++) {
            th::Tensor cur_controlnet_res_torch =
                torch::empty(th::IntArrayRef(controlnet_output_sizes[i]),
                             torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
            lyradiff::Tensor cur_output = convert_tensor<T>(cur_controlnet_res_torch);

            output_th_tensors.push_back(cur_controlnet_res_torch);
            output_tensors.push_back(cur_output);
        }

        lyradiff::Tensor hidden_states_tensor         = convert_tensor<T>(hidden_states);
        lyradiff::Tensor encoder_hidden_states_tensor = convert_tensor<T>(encoder_hidden_states);
        lyradiff::Tensor conditioning_img_tensor      = convert_tensor<T>(conditioning_img);
        lyradiff::Tensor aug_emb_tensor               = convert_tensor<T>(controlnet_aug);

        TensorMap input_tensor({{"hidden_states", hidden_states_tensor},
                                {"encoder_hidden_states", encoder_hidden_states_tensor},
                                {"conditioning_img", conditioning_img_tensor}});
        TensorMap add_tensor({{"aug_emb", aug_emb_tensor}});
        model_->controlnet_forward(output_tensors,
                                   &input_tensor,
                                   &add_tensor,
                                   timestep,
                                   loaded_controlnet_weights[controlnet_name],
                                   input_controlnet_scales);
        cudaDeviceSynchronize();
        lock_->unlock();

        return output_th_tensors;
    }

    th::Tensor unet_forward(th::Tensor                                      hidden_states,
                            th::Tensor                                      encoder_hidden_states,
                            th::Tensor                                      aug_emb,
                            const double                                    timestep,
                            th::optional<vector<th::Tensor>>                controlnet_results,
                            th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors,
                            th::optional<th::Dict<std::string, double>>     scale_params) override
    {
        size_t                        batch_size = hidden_states.size(0);
        size_t                        height     = hidden_states.size(1);
        size_t                        width      = hidden_states.size(2);
        std::vector<lyradiff::Tensor> controlnet_results_lyra_tensors;
        if (controlnet_results.has_value()) {
            std::vector<th::Tensor> controlnet_results_value = controlnet_results.value();
            for (int i = 0; i < controlnet_results_value.size(); i++) {
                controlnet_results_lyra_tensors.push_back(convert_tensor<T>(controlnet_results_value[i]));
            }
        }
        // cudaStreamSynchronize(stream_);
        th::Tensor res_hidden_state_torch =
            torch::empty({batch_size, height, width, output_channels_},
                         torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
        lyradiff::Tensor output                       = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor         = convert_tensor<T>(hidden_states);
        lyradiff::Tensor encoder_hidden_states_tensor = convert_tensor<T>(encoder_hidden_states);
        lyradiff::Tensor aug_emb_tensor               = convert_tensor<T>(aug_emb);

        TensorMap input_tensor(
            {{"hidden_states", hidden_states_tensor}, {"encoder_hidden_states", encoder_hidden_states_tensor}});
        TensorMap add_tensor({{"aug_emb", aug_emb_tensor}});
        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);

        insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        insertParamsToMap(input_tensor, scale_params);

        lock_->lock();
        model_->unet_forward(
            &output_tensor, &input_tensor, &add_tensor, timestep, model_weight_, controlnet_results_lyra_tensors);
        cudaDeviceSynchronize();
        lock_->unlock();

        // cout << "unet before return res_hidden_state_torch" << endl;

        return res_hidden_state_torch;
    }

    void load_lora(const std::string lora_path,
                   const std::string lora_name,
                   const double      lora_strength,
                   const std::string model_dtype) override
    {
        if (loaded_lora_strength.find(lora_name) != loaded_lora_strength.end()) {
            throw "lora is already loaded, please unload first";
        }

        lyradiff::FtCudaDataType model_file_type = lyradiff::FtCudaDataType::FP32;

        if (model_dtype == "fp32") {
            model_file_type = lyradiff::FtCudaDataType::FP32;
        }
        else if (model_dtype == "fp16") {
            model_file_type = lyradiff::FtCudaDataType::FP16;
            cout << "model model_dtype " << model_dtype << endl;
        }
        else {
            throw "wrong model_dtype";
        }

        if (loaded_lora.find(lora_name) != loaded_lora.end()) {
            lock_->lock();
            model_weight_->loadLoraFromCache("", loaded_lora[lora_name], lora_strength, false);
            cudaDeviceSynchronize();
            loaded_lora_strength[lora_name] = lora_strength;
            lock_->unlock();
        }
        else {
            lock_->lock();
            std::unordered_map<std::string, T*> cur_map;
            model_weight_->loadLoraFromWeight(lora_path, "", cur_map, lora_strength, model_file_type, stream_);
            cudaDeviceSynchronize();
            loaded_lora[lora_name]          = cur_map;
            loaded_lora_strength[lora_name] = lora_strength;
            lock_->unlock();
        }
    }

    void load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights, const double lora_strength) override
    {
        lock_->lock();

        std::unordered_map<std::string, T*> tmp_weights;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            tmp_weights[it->key()] = reinterpret_cast<T*>(it->value().data_ptr());
        }

        model_weight_->loadLoraFromCache("", tmp_weights, lora_strength, true);
        lock_->unlock();
    }

    void load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength)
    {
        lock_->lock();
        model_weight_->loadLoraByName(lora_name, 0, reinterpret_cast<T*>(lora_weights.data_ptr()), lora_strength);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void unload_lora(const std::string lora_name, const bool clean_mem_cache) override
    {
        if (loaded_lora_strength.find(lora_name) == loaded_lora_strength.end()) {
            throw "lora is not loaded";
        }

        lock_->lock();
        model_weight_->loadLoraFromCache("", loaded_lora[lora_name], -1.0 * loaded_lora_strength[lora_name], false);
        cudaDeviceSynchronize();
        loaded_lora_strength.erase(lora_name);

        // 如果需要立马 clean 掉这块的 cache，就直接从本地删除
        if (clean_mem_cache) {
            for (auto it = loaded_lora[lora_name].begin(); it != loaded_lora[lora_name].end(); it++) {
                free(it->second);
            }

            loaded_lora.erase(lora_name);
        }
        lock_->unlock();
    }

    void clean_lora_cache() override
    {
        for (auto it = loaded_lora.begin(); it != loaded_lora.end();) {
            // 只有没有被 load 的 lora 才能被
            if (loaded_lora_strength.find(it->first) == loaded_lora_strength.end()) {

                for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
                    free(it2->second);
                }

                it = loaded_lora.erase(it);
            }
            else {
                it++;
            }
        }
    }

    virtual void load_ip_adapter(const std::string& ip_adapter_path,
                                 const std::string& ip_adapter_name,
                                 const double&      scale,
                                 const std::string& model_dtype)
    {
        std::lock_guard<std::mutex> lock(*lock_);
        lyradiff::FtCudaDataType    model_file_type = parse_model_dtype_str(model_dtype);

        model_weight_->loadIPAdapterFromWeight(ip_adapter_path, scale, model_file_type);
        cudaDeviceSynchronize();
    }

    virtual void unload_ip_adapter(const std::string& ip_adapter_name)
    {
        std::lock_guard<std::mutex> lock(*lock_);
        model_weight_->unLoadIPAdapter();
        cudaDeviceSynchronize();
    }

    void reload_unet_model(const std::string model_path, const std::string model_dtype) override
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
            throw std::invalid_argument("wrong model_dtype");
        }

        model_weight_->loadWeights(model_path, model_file_type);

        lock_->unlock();
    }

    void reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device) override
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

        model_weight_->loadWeightsFromCache("", tmp_weights, memcpyKind);
        lock_->unlock();
    }

    void load_controlnet_model(const std::string controlnet_name,
                               const std::string model_path,
                               const std::string model_dtype,
                               const std::string controlnet_mode)
    {
        if (loaded_controlnet_weights.find(controlnet_name) != loaded_controlnet_weights.end()) {
            cout << "controlnet: " << controlnet_name << " already loaded";
            return;
        }

        if (loaded_controlnet_weights.size() >= max_controlnet_count_) {
            throw "cached controlnet model full";
        }

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

        lock_->lock();

        if (cache_controlnet_weights.size() > 0) {
            XLControlNetModelWeight<T>* cur_controlnet = cache_controlnet_weights[cache_controlnet_weights.size() - 1];
            if (cur_controlnet->getControlnetMode() == controlnet_mode) {
                cur_controlnet->loadWeights(model_path, model_file_type);
                loaded_controlnet_weights[controlnet_name] = cur_controlnet;
                cache_controlnet_weights.pop_back();
            }
            else {
                delete cur_controlnet;
                cache_controlnet_weights.pop_back();

                cur_controlnet = new XLControlNetModelWeight<T>(controlnet_mode);
                cur_controlnet->loadWeights(model_path, model_file_type);
                loaded_controlnet_weights[controlnet_name] = cur_controlnet;
            }
        }
        else {
            XLControlNetModelWeight<T>* cur_controlnet = new XLControlNetModelWeight<T>(controlnet_mode);
            cur_controlnet->loadWeights(model_path, model_file_type);
            loaded_controlnet_weights[controlnet_name] = cur_controlnet;
        }
        lock_->unlock();
    }

    void load_controlnet_model_from_state_dict(const std::string                 controlnet_name,
                                               th::Dict<std::string, th::Tensor> weights,
                                               const std::string                 device,
                                               const std::string                 controlnet_mode)
    {
        if (loaded_controlnet_weights.find(controlnet_name) != loaded_controlnet_weights.end()) {
            cout << "controlnet: " << controlnet_name << " already loaded";
            return;
        }

        if (loaded_controlnet_weights.size() >= max_controlnet_count_) {
            throw "cached controlnet model full";
        }

        lock_->lock();

        std::unordered_map<std::string, void*> tmp_weights;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            tmp_weights[it->key()] = reinterpret_cast<void*>(it->value().data_ptr());
        }

        cudaMemcpyKind memcpyKind = cudaMemcpyHostToDevice;
        if (device.rfind("cuda", 0) == 0) {
            memcpyKind = cudaMemcpyDeviceToDevice;
        }

        if (cache_controlnet_weights.size() > 0) {
            XLControlNetModelWeight<T>* cur_controlnet = cache_controlnet_weights[cache_controlnet_weights.size() - 1];
            if (cur_controlnet->getControlnetMode() == controlnet_mode) {
                cur_controlnet->loadWeightsFromCache("", tmp_weights, memcpyKind);
                loaded_controlnet_weights[controlnet_name] = cur_controlnet;
                cache_controlnet_weights.pop_back();
            }
            else {
                delete cur_controlnet;
                cache_controlnet_weights.pop_back();

                cur_controlnet = new XLControlNetModelWeight<T>(controlnet_mode);
                cur_controlnet->loadWeightsFromCache("", tmp_weights, memcpyKind);
                loaded_controlnet_weights[controlnet_name] = cur_controlnet;
            }
        }
        else {
            XLControlNetModelWeight<T>* cur_controlnet = new XLControlNetModelWeight<T>(controlnet_mode);
            cur_controlnet->loadWeightsFromCache("", tmp_weights, memcpyKind);
            loaded_controlnet_weights[controlnet_name] = cur_controlnet;
        }
        lock_->unlock();
    }

    void unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache)
    {
        if (loaded_controlnet_weights.find(controlnet_name) == loaded_controlnet_weights.end()) {
            throw "cannot unload a not loaded model";
        }

        lock_->lock();
        if (clean_mem_cache) {
            delete loaded_controlnet_weights[controlnet_name];
            loaded_controlnet_weights.erase(controlnet_name);
        }
        else {
            cache_controlnet_weights.push_back(loaded_controlnet_weights[controlnet_name]);
            loaded_controlnet_weights.erase(controlnet_name);
        }
        lock_->unlock();
    }

    std::vector<std::string> get_loaded_controlnet()
    {

        std::vector<std::string> res;
        lock_->lock();
        for (const auto& element : loaded_controlnet_weights) {
            res.push_back(element.first);
        }
        lock_->unlock();
        return res;
    }

    std::vector<std::string> get_loaded_lora()
    {
        std::vector<std::string> res;
        lock_->lock();
        for (const auto& element : loaded_lora_strength) {
            res.push_back(element.first);
        }
        lock_->unlock();
        return res;
    }

    void clean_controlnet_cache()
    {
        lock_->lock();
        for (int i = 0; i < cache_controlnet_weights.size(); i++) {
            delete cache_controlnet_weights[i];
            cache_controlnet_weights[i] = nullptr;
        }

        cache_controlnet_weights = std::vector<XLControlNetModelWeight<T>*>();
        lock_->unlock();
    }
};

class XLUnet2dConditionalModelOp: public th::jit::CustomClassHolder {
public:
    XLUnet2dConditionalModelOp(const std::string inference_dtype,
                               const int64_t     input_channels,
                               const int64_t     output_channels,
                               const int64_t     quant_level);
    ~XLUnet2dConditionalModelOp();

    th::Tensor forward(th::Tensor                                      hidden_states,
                       th::Tensor                                      encoder_hidden_states,
                       const double                                    timestep,
                       th::Tensor&                                     aug_emb,
                       th::optional<vector<std::string>>               controlnet_names      = th::nullopt,
                       th::optional<vector<th::Tensor>>                controlnet_conds      = th::nullopt,
                       th::optional<vector<th::Tensor>>                controlnet_augs       = th::nullopt,
                       th::optional<vector<vector<double>>>            controlnet_scales     = th::nullopt,
                       th::optional<bool>                              controlnet_guess_mode = th::nullopt,
                       th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors     = th::nullopt,
                       th::optional<th::Dict<std::string, double>>     scale_params          = th::nullopt);

    std::vector<th::Tensor> controlnet_forward(th::Tensor          hidden_states,
                                               th::Tensor          encoder_hidden_states,
                                               th::Tensor          conditioning_img,
                                               th::Tensor          controlnet_aug,
                                               const double        timestep,
                                               std::vector<double> controlnet_scales,
                                               std::string         controlnet_name);

    th::Tensor unet_forward(th::Tensor                                      hidden_states,
                            th::Tensor                                      encoder_hidden_states,
                            th::Tensor                                      aug_emb,
                            const double                                    timestep,
                            th::optional<vector<th::Tensor>>                controlnet_results = th::nullopt,
                            th::optional<th::Dict<std::string, th::Tensor>> map_extra_tensors  = th::nullopt,
                            th::optional<th::Dict<std::string, double>>     scale_params       = th::nullopt);

    void load_lora(const std::string lora_path,
                   const std::string lora_name,
                   const double      lora_strength,
                   const std::string model_dtype);

    void load_lora_from_state_dict(th::Dict<std::string, th::Tensor> weights, const double lora_strength);

    void load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength);

    void unload_lora(const std::string lora_name, const bool clean_mem_cache);

    void reload_unet_model(const std::string model_path, const std::string model_dtype);

    void reload_unet_model_from_cache(th::Dict<std::string, th::Tensor> weights, const std::string device);

    void load_controlnet_model(const std::string controlnet_name,
                               const std::string model_path,
                               const std::string model_dtype,
                               const std::string controlnet_mode = "large");

    void load_controlnet_model_from_state_dict(const std::string                 controlnet_name,
                                               th::Dict<std::string, th::Tensor> weights,
                                               const std::string                 device,
                                               const std::string                 controlnet_mode = "large");

    void unload_controlnet_model(const std::string controlnet_name, const bool clean_mem_cache);

    void clean_controlnet_cache();

    void clean_lora_cache();

    void load_ip_adapter(const std::string& ip_adapter_path,
                         const std::string& ip_adapter_name,
                         const double&      scale,
                         const std::string& model_dtype);

    void unload_ip_adapter(const std::string& ip_adapter_name);

    std::vector<std::string> get_loaded_lora();

    std::vector<std::string> get_loaded_controlnet();

private:
    // const at::ScalarType    st_;
    IFUnet*     model_;
    std::string inference_dtype_;
};

}  // namespace torch_ext
