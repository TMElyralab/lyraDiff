#include "ZeroSFTWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ZeroSFTWeight<T>::ZeroSFTWeight(size_t project_channels, size_t cond_channels, size_t concat_channels)
{
    project_channels_ = project_channels;
    cond_channels_    = cond_channels;
    concat_channels_  = concat_channels;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ZeroSFTWeight<T>::~ZeroSFTWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(mlp_conv_weight);
        deviceFree(mlp_conv_bias);
        deviceFree(zero_conv_weight);
        deviceFree(zero_conv_bias);
        deviceFree(zero_mul_weight);
        deviceFree(zero_mul_bias);
        deviceFree(zero_add_weight);
        deviceFree(zero_add_bias);
        deviceFree(norm_gamma);
        deviceFree(norm_beta);
    }
}

template<typename T>
void ZeroSFTWeight<T>::mallocWeights()
{
    if (!is_maintain_buffer) {
        deviceMalloc(&mlp_conv_weight, project_channels_ * 3 * 3 * nhidden);
        deviceMalloc(&mlp_conv_bias, nhidden);

        deviceMalloc(&zero_conv_weight, cond_channels_ * 1 * 1 * project_channels_);
        deviceMalloc(&zero_conv_bias, project_channels_);

        deviceMalloc(&zero_mul_weight, (cond_channels_ + concat_channels_) * 3 * 3 * nhidden);
        deviceMalloc(&zero_mul_bias, cond_channels_ + concat_channels_);

        deviceMalloc(&zero_add_weight, (cond_channels_ + concat_channels_) * 3 * 3 * nhidden);
        deviceMalloc(&zero_add_bias, cond_channels_ + concat_channels_);

        deviceMalloc(&norm_gamma, cond_channels_ + concat_channels_);
        deviceMalloc(&norm_beta, cond_channels_ + concat_channels_);

        is_maintain_buffer = true;
    }
}

template<typename T>
void ZeroSFTWeight<T>::loadWeights(const std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    // cout << "cur ZeroSFTWeight prefix " << prefix << endl;

    loadWeightFromBin<T>(
        mlp_conv_weight, {project_channels_ * 3 * 3 * nhidden}, prefix + "mlp_shared.0.weight.bin", model_file_type);
    loadWeightFromBin<T>(mlp_conv_bias, {nhidden}, prefix + "mlp_shared.0.bias.bin", model_file_type);

    loadWeightFromBin<T>(zero_conv_weight,
                         {cond_channels_ * 1 * 1 * project_channels_},
                         prefix + "zero_conv.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(zero_conv_bias, {project_channels_}, prefix + "zero_conv.bias.bin", model_file_type);

    loadWeightFromBin<T>(zero_mul_weight,
                         {(cond_channels_ + concat_channels_) * 3 * 3 * nhidden},
                         prefix + "zero_mul.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        zero_mul_bias, {cond_channels_ + concat_channels_}, prefix + "zero_mul.bias.bin", model_file_type);

    loadWeightFromBin<T>(zero_add_weight,
                         {(cond_channels_ + concat_channels_) * 3 * 3 * nhidden},
                         prefix + "zero_add.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        zero_add_bias, {cond_channels_ + concat_channels_}, prefix + "zero_add.bias.bin", model_file_type);

    loadWeightFromBin<T>(
        norm_gamma, {cond_channels_ + concat_channels_}, prefix + "param_free_norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(
        norm_beta, {cond_channels_ + concat_channels_}, prefix + "param_free_norm.bias.bin", model_file_type);
}

template<typename T>
void ZeroSFTWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                            std::unordered_map<std::string, void*>& weights,
                                            cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_mlp_shared_weights      = weights[prefix + "mlp_shared.0.weight"];
    void* tmp_mlp_shared_bias         = weights[prefix + "mlp_shared.0.bias"];
    void* tmp_zero_conv_weights       = weights[prefix + "zero_conv.weight"];
    void* tmp_zero_conv_bias          = weights[prefix + "zero_conv.bias"];
    void* tmp_zero_mul_weights        = weights[prefix + "zero_mul.weight"];
    void* tmp_zero_mul_bias           = weights[prefix + "zero_mul.bias"];
    void* tmp_zero_add_weights        = weights[prefix + "zero_add.weight"];
    void* tmp_zero_add_bias           = weights[prefix + "zero_add.bias"];
    void* tmp_param_free_norm_weights = weights[prefix + "param_free_norm.weight"];
    void* tmp_param_free_norm_bias    = weights[prefix + "param_free_norm.bias"];

    cudaMemcpy(mlp_conv_weight, tmp_mlp_shared_weights, sizeof(T) * project_channels_ * 3 * 3 * nhidden, memcpyKind);
    cudaMemcpy(mlp_conv_bias, tmp_mlp_shared_bias, sizeof(T) * nhidden, memcpyKind);

    cudaMemcpy(zero_conv_weight,
               tmp_zero_conv_weights,
               sizeof(T) * cond_channels_ * 1 * 1 * project_channels_,
               memcpyKind);
    cudaMemcpy(zero_conv_bias, tmp_zero_conv_bias, sizeof(T) * project_channels_, memcpyKind);

    cudaMemcpy(zero_mul_weight,
               tmp_zero_mul_weights,
               sizeof(T) * (cond_channels_ + concat_channels_) * 3 * 3 * nhidden,
               memcpyKind);
    cudaMemcpy(zero_mul_bias, tmp_zero_mul_bias, sizeof(T) * (cond_channels_ + concat_channels_), memcpyKind);

    cudaMemcpy(zero_add_weight,
               tmp_zero_add_weights,
               sizeof(T) * (cond_channels_ + concat_channels_) * 3 * 3 * nhidden,
               memcpyKind);
    cudaMemcpy(zero_add_bias, tmp_zero_add_bias, sizeof(T) * (cond_channels_ + concat_channels_), memcpyKind);

    cudaMemcpy(norm_gamma, tmp_param_free_norm_weights, sizeof(T) * (cond_channels_ + concat_channels_), memcpyKind);
    cudaMemcpy(norm_beta, tmp_param_free_norm_bias, sizeof(T) * (cond_channels_ + concat_channels_), memcpyKind);
}

template class ZeroSFTWeight<float>;
template class ZeroSFTWeight<half>;
}  // namespace lyradiff