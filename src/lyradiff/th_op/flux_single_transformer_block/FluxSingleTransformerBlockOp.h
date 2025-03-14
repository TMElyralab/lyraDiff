#pragma once

#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerFP8Block.h"
#include "src/lyradiff/layers/flux_single_transformer_block/FluxSingleTransformerInt4Block.h"
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

class IFSingleTransformerBlock {
public:
    virtual ~IFSingleTransformerBlock() {}

    virtual th::Tensor forward(th::Tensor hidden_states, th::Tensor rope_emb, th::Tensor temb_emb) = 0;

    virtual void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context) = 0;

    virtual void
    load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength) = 0;

    virtual void clear_lora() = 0;

    virtual void reload_model_from_bin(const std::string model_path, const std::string model_dtype) = 0;

    virtual void reload_model_from_state_dict(const std::string                 prefix,
                                              th::Dict<std::string, th::Tensor> weights,
                                              const std::string                 device) = 0;
};

template<typename T>
class LyraDiffFluxSingleTransformer: public IFSingleTransformerBlock {

private:
    std::mutex* lock_;

    FluxSingleTransformerBlock<T>*       model_        = nullptr;
    FluxSingleTransformerBlockWeight<T>* model_weight_ = nullptr;

    size_t embedding_head_dim_ = 128;
    size_t embedding_head_num_ = 24;
    size_t embedding_dim_      = embedding_head_dim_ * embedding_head_num_;
    size_t mlp_scale_          = 4;

    int64_t       quant_level_      = 0;
    LyraQuantType quant_level_enum_ = LyraQuantType::NONE;

public:
    LyraDiffFluxSingleTransformer(size_t       embedding_dim,
                                  size_t       embedding_head_num,
                                  size_t       embedding_head_dim,
                                  size_t       mlp_scale,
                                  const size_t quant_level = 0)
    {
        // max_controlnet_count_ = max_controlnet_count;
        quant_level_        = quant_level;
        embedding_head_num_ = embedding_head_num;
        embedding_head_dim_ = embedding_head_dim;
        embedding_dim_      = embedding_dim;
        mlp_scale_          = mlp_scale;

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

        lock_ = new std::mutex();
    }

    ~LyraDiffFluxSingleTransformer() override
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
        if (quant_level_enum_ == LyraQuantType::NONE) {
            model_ = new FluxSingleTransformerBlock<T>(embedding_dim_,
                                                       embedding_head_num_,
                                                       embedding_head_dim_,
                                                       mlp_scale_,
                                                       lyradiff_context->stream_,
                                                       lyradiff_context->cublas_wrapper_,
                                                       lyradiff_context->allocator_,
                                                       false,
                                                       false,
                                                       quant_level_enum_);

            model_weight_ = new FluxSingleTransformerBlockWeight<T>(embedding_dim_,
                                                                    embedding_head_num_,
                                                                    embedding_head_dim_,
                                                                    mlp_scale_,
                                                                    quant_level_enum_,
                                                                    lyradiff_context->allocator_);
        }
        else if (quant_level_enum_ == LyraQuantType::FP8_W8A8_FULL || quant_level_enum_ == LyraQuantType::FP8_W8A8) {
            model_ = new FluxSingleTransformerFP8Block<T>(embedding_dim_,
                                                          embedding_head_num_,
                                                          embedding_head_dim_,
                                                          mlp_scale_,
                                                          lyradiff_context->stream_,
                                                          lyradiff_context->cublas_wrapper_,
                                                          lyradiff_context->allocator_,
                                                          false,
                                                          false,
                                                          quant_level_enum_);

            model_weight_ = new FluxSingleTransformerFP8BlockWeight<T>(embedding_dim_,
                                                                       embedding_head_num_,
                                                                       embedding_head_dim_,
                                                                       mlp_scale_,
                                                                       quant_level_enum_,
                                                                       lyradiff_context->allocator_);
        }
        else {
            model_ = new FluxSingleTransformerInt4Block<T>(embedding_dim_,
                                                           embedding_head_num_,
                                                           embedding_head_dim_,
                                                           mlp_scale_,
                                                           lyradiff_context->stream_,
                                                           lyradiff_context->cublas_wrapper_,
                                                           lyradiff_context->allocator_,
                                                           false,
                                                           false,
                                                           quant_level_enum_);

            model_weight_ = new FluxSingleTransformerInt4BlockWeight<T>(embedding_dim_,
                                                                        embedding_head_num_,
                                                                        embedding_head_dim_,
                                                                        mlp_scale_,
                                                                        quant_level_enum_,
                                                                        lyradiff_context->allocator_);
        }
    }

    th::Tensor forward(th::Tensor hidden_states, th::Tensor rope_emb, th::Tensor temb_emb)
    {
        size_t batch_size = hidden_states.size(0);
        size_t seq_len    = hidden_states.size(1);
        size_t dim        = hidden_states.size(2);

        // cudaStreamSynchronize(stream_);
        th::Tensor res_hidden_state_torch = torch::empty(
            {batch_size, seq_len, dim}, torch::dtype(hidden_states.dtype()).device(torch::kCUDA).requires_grad(false));

        lyradiff::Tensor output               = convert_tensor<T>(res_hidden_state_torch);
        lyradiff::Tensor hidden_states_tensor = convert_tensor<T>(hidden_states);
        lyradiff::Tensor rope_emb_tensor      = convert_tensor<T>(rope_emb);
        lyradiff::Tensor temb_emb_tensor      = convert_tensor<T>(temb_emb);

        TensorMap input_tensor(
            {{"input", hidden_states_tensor}, {"temb", temb_emb_tensor}, {"rope_emb", rope_emb_tensor}});

        TensorMap output_tensor({{"output", output}});

        LyraDiffContext context;
        input_tensor.setContext(&context);

        // insertTorchTensorMapToMapContext<T>(input_tensor, map_extra_tensors);
        // insertParamsToMap(input_tensor, scale_params);

        lock_->lock();
        if (quant_level_enum_ == LyraQuantType::NONE) {
            model_->forward(&output_tensor, &input_tensor, model_weight_);
        }
        else if (quant_level_enum_ == LyraQuantType::FP8_W8A8_FULL || quant_level_enum_ == LyraQuantType::FP8_W8A8) {
            FluxSingleTransformerFP8Block<T>* block = (FluxSingleTransformerFP8Block<T>*)model_;

            block->forward(&output_tensor, &input_tensor, (FluxSingleTransformerFP8BlockWeight<T>*)model_weight_);
        }
        else {
            FluxSingleTransformerInt4Block<T>* block = (FluxSingleTransformerInt4Block<T>*)model_;
            block->forward(&output_tensor, &input_tensor, (FluxSingleTransformerInt4BlockWeight<T>*)model_weight_);
        }
        cudaDeviceSynchronize();
        lock_->unlock();

        return res_hidden_state_torch;
    }

    void reload_model_from_bin(const std::string model_path, const std::string model_dtype) override
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

    void reload_model_from_state_dict(const std::string                 prefix,
                                      th::Dict<std::string, th::Tensor> weights,
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

        model_weight_->loadWeightsFromCache(prefix, tmp_weights, memcpyKind);
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

class FluxSingleTransformerBlockOp: public th::jit::CustomClassHolder {
public:
    FluxSingleTransformerBlockOp(const std::string inference_dtype,
                                 size_t            embedding_dim,
                                 size_t            embedding_head_num,
                                 size_t            embedding_head_dim,
                                 size_t            mlp_scale,
                                 const size_t      quant_level = 0);
    ~FluxSingleTransformerBlockOp();

    void initialize(const c10::intrusive_ptr<torch_ext::LyraDiffCommonContext>& lyradiff_context);

    th::Tensor forward(th::Tensor hidden_states, th::Tensor rope_emb, th::Tensor temb_emb);

    void load_lora_by_name(std::vector<std::string> lora_name, th::Tensor lora_weights, const double lora_strength);

    void clear_lora();

    void reload_model_from_bin(const std::string model_path, const std::string model_dtype);

    void reload_model_from_state_dict(const std::string                 prefix,
                                      th::Dict<std::string, th::Tensor> weights,
                                      const std::string                 device);

private:
    // const at::ScalarType    st_;
    IFSingleTransformerBlock* model_;
    std::string               inference_dtype_;
};

}  // namespace torch_ext
