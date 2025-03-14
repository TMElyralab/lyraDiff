#pragma once

#include "src/lyradiff/models/flux_controlnet_model/FluxControlnetModel.h"
#include "src/lyradiff/models/flux_transformer_2d_model/FluxTransformer2DModel.h"
#include "src/lyradiff/th_op/lyradiff_common_context/LyraDiffCommonContext.h"
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

class IFTransformer {
public:
    virtual ~IFTransformer() {}

    virtual th::Tensor transformer_forward(th::Tensor                       hidden_states,
                                           th::Tensor                       encoder_hidden_states,
                                           th::Tensor                       rope_emb,
                                           th::Tensor                       pooled_projection,
                                           const double                     timestep,
                                           const double                     guidance,
                                           th::optional<vector<th::Tensor>> controlnet_block_samples,
                                           th::optional<vector<th::Tensor>> controlnet_single_block_samples,
                                           th::optional<bool>               controlnet_blocks_repeat) = 0;

    virtual void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context) = 0;

    virtual void
    load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength) = 0;

    virtual void clear_lora() = 0;

    virtual void reload_transformer_model(const std::string model_path, const std::string model_dtype) = 0;

    virtual void reload_transformer_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                          const std::string                 device) = 0;
};

template<typename T>
class LyraDiffFluxTransformer2DModel: public IFTransformer {

private:
    std::mutex* lock_;

    FluxTransformer2DModel<T>*       model_        = nullptr;
    FluxTransformer2DModelWeight<T>* model_weight_ = nullptr;

    FluxControlnetModel<T>*                                        controlnet_model_ = nullptr;  // deprecated
    std::unordered_map<std::string, FluxControlnetModelWeight<T>*> loaded_controlnet_weights;    // deprecated

    std::string machine_id;
    std::string token_;

    bool loop_ = true;

    bool can_use = false;

    int retry_limit = 5;  // 暂时auth 给5次使用机会

    size_t input_channels_        = 64;
    size_t output_channels_       = 64;
    size_t num_layers_            = 19;
    size_t num_single_layers_     = 38;
    size_t attention_head_dim_    = 128;
    size_t num_attention_heads_   = 24;
    size_t pooled_projection_dim_ = 768;
    size_t joint_attention_dim_   = 4096;
    bool   guidance_embeds_       = true;

    int64_t       quant_level_      = 0;
    LyraQuantType quant_level_enum_ = LyraQuantType::NONE;

    bool is_internal_ = false;

public:
    LyraDiffFluxTransformer2DModel(const size_t input_channels        = 64,
                                   const size_t num_layers            = 19,
                                   const size_t num_single_layers     = 38,
                                   const size_t attention_head_dim    = 128,
                                   const size_t num_attention_heads   = 24,
                                   const size_t pooled_projection_dim = 768,
                                   const size_t joint_attention_dim   = 4096,
                                   const bool   guidance_embeds       = true,
                                   const size_t quant_level           = 0)
    {
        // max_controlnet_count_ = max_controlnet_count;
        quant_level_           = quant_level;
        num_layers_            = num_layers;
        num_single_layers_     = num_single_layers;
        attention_head_dim_    = attention_head_dim;
        num_attention_heads_   = num_attention_heads;
        pooled_projection_dim_ = pooled_projection_dim;
        joint_attention_dim_   = joint_attention_dim;
        guidance_embeds_       = guidance_embeds;

        quant_level_enum_ = LyraQuantType::NONE;
        if (quant_level == 1) {
            quant_level_enum_ = LyraQuantType::FP8_W8A8;
        }
        else if (quant_level == 2) {
            quant_level_enum_ = LyraQuantType::FP8_W8A8_FULL;
        }
        else if (quant_level == 7) {
            quant_level_enum_ = LyraQuantType::INT4_W4A4;
        }
        else if (quant_level == 8) {
            quant_level_enum_ = LyraQuantType::INT4_W4A4_FULL;
        }

        // cublas_wrapper_mutex_ = new std::mutex();

        lock_ = new std::mutex();

        input_channels_  = input_channels;
        output_channels_ = input_channels_;
    }

    ~LyraDiffFluxTransformer2DModel() override
    {
        if (model_ != nullptr) {
            delete model_;
        }
        if (model_weight_ != nullptr) {
            delete model_weight_;
        }

        delete lock_;
    }

    void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context) override
    {

        model_ = new FluxTransformer2DModel<T>(lyradiff_context->stream_,
                                               lyradiff_context->cublas_wrapper_,
                                               lyradiff_context->allocator_,
                                               false,
                                               false,
                                               input_channels_,
                                               num_layers_,
                                               num_single_layers_,
                                               attention_head_dim_,
                                               num_attention_heads_,
                                               pooled_projection_dim_,
                                               joint_attention_dim_,
                                               guidance_embeds_,
                                               quant_level_enum_);

        model_weight_ = new FluxTransformer2DModelWeight<T>(input_channels_,
                                                            num_layers_,
                                                            num_single_layers_,
                                                            attention_head_dim_,
                                                            num_attention_heads_,
                                                            pooled_projection_dim_,
                                                            joint_attention_dim_,
                                                            guidance_embeds_,
                                                            quant_level_enum_,
                                                            lyradiff_context->allocator_);
    }

    th::Tensor transformer_forward(th::Tensor                       hidden_states,
                                   th::Tensor                       encoder_hidden_states,
                                   th::Tensor                       rope_emb,
                                   th::Tensor                       pooled_projection,
                                   const double                     timestep,
                                   const double                     guidance,
                                   th::optional<vector<th::Tensor>> controlnet_block_samples,
                                   th::optional<vector<th::Tensor>> controlnet_single_block_samples,
                                   th::optional<bool>               controlnet_blocks_repeat)
    {
        size_t batch_size = hidden_states.size(0);
        size_t seq_len    = hidden_states.size(1);

        std::vector<lyradiff::Tensor> controlnet_block_samples_tensors;
        std::vector<lyradiff::Tensor> controlnet_single_block_samples_tensors;
        if (controlnet_block_samples.has_value()) {
            std::vector<th::Tensor> controlnet_results_value = controlnet_block_samples.value();
            for (int i = 0; i < controlnet_results_value.size(); i++) {
                controlnet_block_samples_tensors.push_back(convert_tensor<T>(controlnet_results_value[i]));
            }
        }

        if (controlnet_single_block_samples.has_value()) {
            std::vector<th::Tensor> controlnet_results_value = controlnet_single_block_samples.value();
            for (int i = 0; i < controlnet_results_value.size(); i++) {
                controlnet_single_block_samples_tensors.push_back(convert_tensor<T>(controlnet_results_value[i]));
            }
        }

        bool controlnet_blocks_repeat_v = false;
        if (controlnet_blocks_repeat.has_value()) {
            controlnet_blocks_repeat_v = controlnet_blocks_repeat.value();
        }

        // cudaStreamSynchronize(stream_);
        th::Tensor res_hidden_state_torch =
            torch::empty({batch_size, seq_len, output_channels_},
                         torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
        lyradiff::Tensor output                       = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor         = convert_tensor<T>(hidden_states);
        lyradiff::Tensor encoder_hidden_states_tensor = convert_tensor<T>(encoder_hidden_states);
        lyradiff::Tensor rope_emb_tensor              = convert_tensor<T>(rope_emb);
        lyradiff::Tensor pooled_projection_tensor     = convert_tensor<T>(pooled_projection);

        TensorMap input_tensor({{"hidden_states", hidden_states_tensor},
                                {"encoder_hidden_states", encoder_hidden_states_tensor},
                                {"pooled_projection", pooled_projection_tensor},
                                {"rope_embs", rope_emb_tensor}});

        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);

        // insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        // insertParamsToMap(input_tensor, scale_params);

        lock_->lock();
        model_->transformer_forward(&output_tensor,
                                    &input_tensor,
                                    timestep,
                                    guidance,
                                    model_weight_,
                                    controlnet_block_samples_tensors,
                                    controlnet_single_block_samples_tensors,
                                    controlnet_blocks_repeat_v);
        cudaDeviceSynchronize();
        lock_->unlock();

        return res_hidden_state_torch;
    }

    void reload_transformer_model(const std::string model_path, const std::string model_dtype) override
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

    void reload_transformer_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                  const std::string                 device) override
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

    void load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength)
    {
        lock_->lock();
        model_weight_->loadLoraByName(lora_name, 0, reinterpret_cast<T*>(lora_weights.data_ptr()), lora_strength);
        cudaDeviceSynchronize();
        lock_->unlock();
    }

    void clear_lora()
    {
        lock_->lock();
        model_weight_->clear_lora();
        cudaDeviceSynchronize();
        lock_->unlock();
    }
};

class FluxTransformer2DModelOp: public th::jit::CustomClassHolder {
public:
    FluxTransformer2DModelOp(const std::string inference_dtype,
                             const size_t      input_channels        = 64,
                             const size_t      num_layers            = 19,
                             const size_t      num_single_layers     = 38,
                             const size_t      attention_head_dim    = 128,
                             const size_t      num_attention_heads   = 24,
                             const size_t      pooled_projection_dim = 768,
                             const size_t      joint_attention_dim   = 4096,
                             const bool        guidance_embeds       = true,
                             const size_t      quant_level           = 0);
    ~FluxTransformer2DModelOp();

    void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context);

    th::Tensor transformer_forward(th::Tensor                       hidden_states,
                                   th::Tensor                       encoder_hidden_states,
                                   th::Tensor                       rope_emb,
                                   th::Tensor                       pooled_projection,
                                   const double                     timestep,
                                   const double                     guidance,
                                   th::optional<vector<th::Tensor>> controlnet_block_samples        = th::nullopt,
                                   th::optional<vector<th::Tensor>> controlnet_single_block_samples = th::nullopt,
                                   th::optional<bool>               controlnet_blocks_repeat        = false);

    void load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength);

    void clear_lora();

    void reload_transformer_model(const std::string model_path, const std::string model_dtype);

    void reload_transformer_model_from_state_dict(th::Dict<std::string, th::Tensor> weights, const std::string device);

    // void load_ip_adapter(const std::string& ip_adapter_path,
    //                      const std::string& ip_adapter_name,
    //                      const double&      scale,
    //                      const std::string& model_dtype);

    // void unload_ip_adapter(const std::string& ip_adapter_name);

    // void load_facein(const std::string& facein_path, const std::string& model_type);
    // void unload_facein();

    // std::vector<std::string> get_loaded_lora();

    // std::vector<std::string> get_loaded_controlnet();

private:
    // const at::ScalarType    st_;
    IFTransformer* model_;
    std::string    inference_dtype_;
};

}  // namespace torch_ext
