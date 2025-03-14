#pragma once

#include "src/lyradiff/models/flux_controlnet_model/FluxControlnetModel.h"
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

class IFFluxControlnet {
public:
    virtual ~IFFluxControlnet() {}

    virtual std::vector<th::Tensor> controlnet_forward(th::Tensor   hidden_states,
                                                       th::Tensor   encoder_hidden_states,
                                                       th::Tensor   rope_emb,
                                                       th::Tensor   pooled_projection,
                                                       th::Tensor   controlnet_cond,
                                                       const double timestep,
                                                       const double guidance,
                                                       const double controlnet_scale) = 0;

    virtual void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context) = 0;

    virtual void reload_controlnet_model(const std::string model_path,
                                         const size_t      num_layers,
                                         const size_t      num_single_layers,
                                         const std::string model_dtype) = 0;

    virtual void reload_controlnet_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                         const size_t                      num_layers,
                                                         const size_t                      num_single_layers,
                                                         const std::string                 device) = 0;
};

template<typename T>
class LyraDiffFluxControlnetModel: public IFFluxControlnet {

private:
    // cudaStream_t stream_ = cudaStreamDefault;
    // cudaStream_t stream_assistant_;

    // cublasHandle_t   cublas_handle_;
    // cublasLtHandle_t cublaslt_handle_;
    // cudnnHandle_t    cudnn_handle_;

    // cublasAlgoMap*                  cublas_algo_map_;
    // Allocator<AllocatorType::CUDA>* allocator_;
    // std::mutex*                     cublas_wrapper_mutex_;
    // lyradiff::cublasMMWrapper*        cublas_wrapper_;
    std::mutex* lock_;

    FluxControlnetModel<T>*       model_;
    FluxControlnetModelWeight<T>* model_weight_;

    std::string machine_id;
    std::string token_;

    int retry_limit = 5;  // 暂时auth 给5次使用机会

    size_t input_channels_        = 64;
    size_t output_channels_       = 64;
    size_t num_layers_            = 2;
    size_t num_single_layers_     = 0;
    size_t attention_head_dim_    = 128;
    size_t num_attention_heads_   = 24;
    size_t pooled_projection_dim_ = 768;
    size_t joint_attention_dim_   = 4096;
    bool   guidance_embeds_       = true;

    int64_t       quant_level_      = 0;
    LyraQuantType quant_level_enum_ = LyraQuantType::NONE;

public:
    LyraDiffFluxControlnetModel(const size_t input_channels        = 64,
                                const size_t num_layers            = 2,
                                const size_t num_single_layers     = 0,
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

        lock_ = new std::mutex();

        quant_level_enum_ = LyraQuantType::NONE;
        if (quant_level == 1) {
            // quant_level_enum_ = LyraQuantType::FP8_W8A8;
            cout << "flux conrtolnet don't support fp8 quant for now, switch back to no quant" << endl;
        }
        else if (quant_level == 2) {
            // quant_level_enum_ = LyraQuantType::FP8_W8A8_FULL;
            cout << "flux conrtolnet don't support fp8 quant for now, switch back to no quant" << endl;
        }

        input_channels_  = input_channels;
        output_channels_ = input_channels_;
    }

    ~LyraDiffFluxControlnetModel() override
    {
        delete model_;
        delete model_weight_;

        // delete allocator_;
        delete lock_;
    }

    void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context) override
    {
        model_ = new FluxControlnetModel<T>(lyradiff_context->stream_,
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

        model_weight_ = new FluxControlnetModelWeight<T>(input_channels_,
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

    std::vector<th::Tensor> controlnet_forward(th::Tensor   hidden_states,
                                               th::Tensor   encoder_hidden_states,
                                               th::Tensor   rope_emb,
                                               th::Tensor   pooled_projection,
                                               th::Tensor   controlnet_cond,
                                               const double timestep,
                                               const double guidance,
                                               const double controlnet_scale)
    {
        // cudaStreamSynchronize(stream_);

        size_t batch_size = hidden_states.size(0);
        size_t seq_len    = hidden_states.size(1);

        // cudaStreamSynchronize(stream_);
        std::vector<th::Tensor>       res_output_tensors;
        std::vector<lyradiff::Tensor> output_tensors;
        std::vector<lyradiff::Tensor> output_single_tensors;

        size_t num_layers        = model_weight_->getNumLayers();
        size_t num_single_layers = model_weight_->getNumSingleLayers();

        for (int i = 0; i < num_layers; i++) {
            th::Tensor res_torch =
                torch::empty({batch_size, seq_len, attention_head_dim_ * num_attention_heads_},
                             torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
            lyradiff::Tensor cur_tensor = convert_tensor<T>(res_torch);
            output_tensors.push_back(cur_tensor);
            res_output_tensors.push_back(res_torch);
        }

        for (int i = 0; i < num_single_layers; i++) {
            th::Tensor res_torch =
                torch::empty({batch_size, seq_len, attention_head_dim_ * num_attention_heads_},
                             torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));
            lyradiff::Tensor cur_tensor = convert_tensor<T>(res_torch);
            output_single_tensors.push_back(cur_tensor);
            res_output_tensors.push_back(res_torch);
        }

        lyradiff::Tensor hidden_states_tensor         = convert_tensor<T>(hidden_states);
        lyradiff::Tensor encoder_hidden_states_tensor = convert_tensor<T>(encoder_hidden_states);
        lyradiff::Tensor rope_emb_tensor              = convert_tensor<T>(rope_emb);
        lyradiff::Tensor pooled_projection_tensor     = convert_tensor<T>(pooled_projection);
        lyradiff::Tensor controlnet_cond_tensor       = convert_tensor<T>(controlnet_cond);

        TensorMap input_tensor({{"hidden_states", hidden_states_tensor},
                                {"encoder_hidden_states", encoder_hidden_states_tensor},
                                {"pooled_projection", pooled_projection_tensor},
                                {"controlnet_condition", controlnet_cond_tensor},
                                {"rope_embs", rope_emb_tensor}});

        LyraDiffContext context;
        input_tensor.setContext(&context);

        lock_->lock();
        model_->controlnet_forward(
            output_tensors, output_single_tensors, &input_tensor, timestep, guidance, controlnet_scale, model_weight_);

        cudaDeviceSynchronize();
        lock_->unlock();

        return res_output_tensors;
    }

    void reload_controlnet_model(const std::string model_path,
                                 const size_t      num_layers,
                                 const size_t      num_single_layers,
                                 const std::string model_dtype) override
    {
        lock_->lock();

        model_weight_->updateConfig(num_layers, num_single_layers);

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

    void reload_controlnet_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                 const size_t                      num_layers,
                                                 const size_t                      num_single_layers,
                                                 const std::string                 device) override
    {
        lock_->lock();

        model_weight_->updateConfig(num_layers, num_single_layers);
        model_->updateConfig(num_layers, num_single_layers);

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
};

class FluxControlnetModelOp: public th::jit::CustomClassHolder {
public:
    FluxControlnetModelOp(const std::string inference_dtype,
                          const size_t      input_channels        = 64,
                          const size_t      num_layers            = 19,
                          const size_t      num_single_layers     = 38,
                          const size_t      attention_head_dim    = 128,
                          const size_t      num_attention_heads   = 24,
                          const size_t      pooled_projection_dim = 768,
                          const size_t      joint_attention_dim   = 4096,
                          const bool        guidance_embeds       = true,
                          const size_t      quant_level           = 0);
    ~FluxControlnetModelOp();

    void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context);

    std::vector<th::Tensor> controlnet_forward(th::Tensor   hidden_states,
                                               th::Tensor   encoder_hidden_states,
                                               th::Tensor   rope_emb,
                                               th::Tensor   pooled_projection,
                                               th::Tensor   controlnet_cond,
                                               const double timestep,
                                               const double guidance,
                                               const double controlnet_scale);

    void reload_controlnet_model(const std::string model_path,
                                 const int64_t     num_layers,
                                 const int64_t     num_single_layers,
                                 const std::string model_dtype);

    void reload_controlnet_model_from_state_dict(th::Dict<std::string, th::Tensor> weights,
                                                 const int64_t                     num_layers,
                                                 const int64_t                     num_single_layers,
                                                 const std::string                 device);

private:
    // const at::ScalarType    st_;
    IFFluxControlnet* model_;
    std::string       inference_dtype_;
};

}  // namespace torch_ext
