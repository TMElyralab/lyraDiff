#include "VaeDecoderWeight.h"
#include "src/lyradiff/utils/memory_utils.h"
#include "src/lyradiff/utils/Tensor.h"
using namespace std;

namespace lyradiff {
template<typename T>
VaeDecoderWeight<T>::VaeDecoderWeight(const size_t in_channels,
                                      const size_t out_channels,
                                      const size_t norm_num_groups,
                                      const bool   is_upcast):
    in_channels_(in_channels), out_channels_(out_channels), norm_num_groups_(norm_num_groups), is_upcast_(is_upcast)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (std::is_same<T, float>()) {
        is_upcast_ = false;
    }

    unet_mid_block_2d_weight = new UNetMidBlock2DWeight<T>(block_out_channels_[0], 1, norm_num_groups_, temb_channels_);

    up_decoder_block_2d_weight_0 = new UpDecoderBlock2dWeight<T>(
        block_out_channels_[0], block_out_channels_[0], norm_num_groups_, temb_channels_, true);

    up_decoder_block_2d_weight_1 = new UpDecoderBlock2dWeight<T>(
        block_out_channels_[0], block_out_channels_[1], norm_num_groups_, temb_channels_, true);

    if (is_upcast_) {
        upcast_up_decoder_block_2d_weight_2 = new UpDecoderBlock2dWeight<float>(
            block_out_channels_[1], block_out_channels_[2], norm_num_groups_, temb_channels_, true);

        upcast_up_decoder_block_2d_weight_3 = new UpDecoderBlock2dWeight<float>(
            block_out_channels_[2], block_out_channels_[3], norm_num_groups_, temb_channels_, false);
    }
    else {
        up_decoder_block_2d_weight_2 = new UpDecoderBlock2dWeight<T>(
            block_out_channels_[1], block_out_channels_[2], norm_num_groups_, temb_channels_, true);

        up_decoder_block_2d_weight_3 = new UpDecoderBlock2dWeight<T>(
            block_out_channels_[2], block_out_channels_[3], norm_num_groups_, temb_channels_, false);
    }

    // TODO: 加入 resnet 2d block weights
}

template<typename T>
VaeDecoderWeight<T>::~VaeDecoderWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(conv_in_weight);
        deviceFree(conv_in_bias);
        deviceFree(conv_norm_out_gamma);
        deviceFree(conv_norm_out_beta);
        deviceFree(conv_out_weight);
        deviceFree(conv_out_bias);

        delete unet_mid_block_2d_weight;
        delete up_decoder_block_2d_weight_0;
        delete up_decoder_block_2d_weight_1;

        unet_mid_block_2d_weight     = nullptr;
        up_decoder_block_2d_weight_0 = nullptr;
        up_decoder_block_2d_weight_1 = nullptr;

        if (is_upcast_) {
            delete upcast_up_decoder_block_2d_weight_2;
            delete upcast_up_decoder_block_2d_weight_3;
            upcast_up_decoder_block_2d_weight_2 = nullptr;
            upcast_up_decoder_block_2d_weight_3 = nullptr;
        }
        else {
            delete up_decoder_block_2d_weight_2;
            delete up_decoder_block_2d_weight_3;
            up_decoder_block_2d_weight_2 = nullptr;
            up_decoder_block_2d_weight_3 = nullptr;
        }
    }
}

template<typename T>
void VaeDecoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    deviceMalloc(&conv_in_weight, in_channels_ * 3 * 3 * block_out_channels_[0]);
    deviceMalloc(&conv_in_bias, block_out_channels_[0]);

    unet_mid_block_2d_weight->mallocWeights();
    up_decoder_block_2d_weight_0->mallocWeights();
    up_decoder_block_2d_weight_1->mallocWeights();
    deviceMalloc(&conv_out_weight, block_out_channels_[3] * 3 * 3 * out_channels_);
    deviceMalloc(&conv_out_bias, out_channels_);

    if (is_upcast_) {
        deviceMalloc(&upcast_conv_norm_out_gamma, block_out_channels_[3]);
        deviceMalloc(&upcast_conv_norm_out_beta, block_out_channels_[3]);

        upcast_up_decoder_block_2d_weight_2->mallocWeights();
        upcast_up_decoder_block_2d_weight_3->mallocWeights();
    }
    else {
        deviceMalloc(&conv_norm_out_gamma, block_out_channels_[3]);
        deviceMalloc(&conv_norm_out_beta, block_out_channels_[3]);

        up_decoder_block_2d_weight_2->mallocWeights();
        up_decoder_block_2d_weight_3->mallocWeights();
    }

    is_maintain_buffer = true;
}

template<typename T>
void VaeDecoderWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    loadWeightFromBin<T>(conv_in_weight,
                         {in_channels_ * 3 * 3 * block_out_channels_[0]},
                         prefix + "conv_in.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_in_bias, {block_out_channels_[0]}, prefix + "conv_in.bias.bin", model_file_type);

    loadWeightFromBin<T>(conv_out_weight,
                         {block_out_channels_[3] * 3 * 3 * out_channels_},
                         prefix + "conv_out.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(conv_out_bias, {out_channels_}, prefix + "conv_out.bias.bin", model_file_type);

    unet_mid_block_2d_weight->loadWeights(prefix + std::string("mid_block."), model_file_type);
    up_decoder_block_2d_weight_0->loadWeights(prefix + std::string("up_blocks.0."), model_file_type);
    up_decoder_block_2d_weight_1->loadWeights(prefix + std::string("up_blocks.1."), model_file_type);
    if (is_upcast_) {
        loadWeightFromBin<float>(
            upcast_conv_norm_out_gamma, {block_out_channels_[3]}, prefix + "conv_norm_out.weight.bin", model_file_type);
        loadWeightFromBin<float>(
            upcast_conv_norm_out_beta, {block_out_channels_[3]}, prefix + "conv_norm_out.bias.bin", model_file_type);

        upcast_up_decoder_block_2d_weight_2->loadWeights(prefix + std::string("up_blocks.2."), model_file_type);
        upcast_up_decoder_block_2d_weight_3->loadWeights(prefix + std::string("up_blocks.3."), model_file_type);
    }
    else {
        loadWeightFromBin<T>(
            conv_norm_out_gamma, {block_out_channels_[3]}, prefix + "conv_norm_out.weight.bin", model_file_type);
        loadWeightFromBin<T>(
            conv_norm_out_beta, {block_out_channels_[3]}, prefix + "conv_norm_out.bias.bin", model_file_type);

        up_decoder_block_2d_weight_2->loadWeights(prefix + std::string("up_blocks.2."), model_file_type);
        up_decoder_block_2d_weight_3->loadWeights(prefix + std::string("up_blocks.3."), model_file_type);
    }
}

// TODO: 修复 upcast 模式的 loadWeightsFromCache
template<typename T>
void VaeDecoderWeight<T>::loadWeightsFromCache(std::string                          prefix,
                                               std::unordered_map<std::string, void*>& weights,
                                               cudaMemcpyKind                       memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }

    void* tmp_conv_in_weight      = weights[prefix + "conv_in.weight"];
    void* tmp_conv_in_bias        = weights[prefix + "conv_in.bias"];
    void* tmp_conv_norm_out_gamma = weights[prefix + "conv_norm_out.weight"];
    void* tmp_conv_norm_out_beta  = weights[prefix + "conv_norm_out.bias"];
    void* tmp_conv_out_weight     = weights[prefix + "conv_out.weight"];
    void* tmp_conv_out_bias       = weights[prefix + "conv_out.bias"];

    weight_loader_manager_glob->doCudaMemcpy(
        conv_in_weight, tmp_conv_in_weight, sizeof(T) * in_channels_ * 3 * 3 * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv_in_bias, tmp_conv_in_bias, sizeof(T) * block_out_channels_[0], memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(
        conv_out_weight, tmp_conv_out_weight, sizeof(T) * block_out_channels_[3] * 3 * 3 * out_channels_, memcpyKind);
    weight_loader_manager_glob->doCudaMemcpy(conv_out_bias, tmp_conv_out_bias, sizeof(T) * out_channels_, memcpyKind);

    unet_mid_block_2d_weight->loadWeightsFromCache(prefix + std::string("mid_block."), weights, memcpyKind);
    up_decoder_block_2d_weight_0->loadWeightsFromCache(prefix + std::string("up_blocks.0."), weights, memcpyKind);
    up_decoder_block_2d_weight_1->loadWeightsFromCache(prefix + std::string("up_blocks.1."), weights, memcpyKind);

    if (is_upcast_) {
        weight_loader_manager_glob->doCudaMemcpy(upcast_conv_norm_out_gamma, tmp_conv_norm_out_gamma, sizeof(float) * block_out_channels_[3], memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(upcast_conv_norm_out_beta, tmp_conv_norm_out_beta, sizeof(float) * block_out_channels_[3], memcpyKind);

        upcast_up_decoder_block_2d_weight_2->loadWeightsFromCache(prefix + std::string("up_blocks.2."), weights, memcpyKind);
        upcast_up_decoder_block_2d_weight_3->loadWeightsFromCache(prefix + std::string("up_blocks.3."), weights, memcpyKind);
    }
    else {
        weight_loader_manager_glob->doCudaMemcpy(conv_norm_out_gamma, tmp_conv_norm_out_gamma, sizeof(T) * block_out_channels_[3], memcpyKind);
        weight_loader_manager_glob->doCudaMemcpy(conv_norm_out_beta, tmp_conv_norm_out_beta, sizeof(T) * block_out_channels_[3], memcpyKind);
        up_decoder_block_2d_weight_2->loadWeightsFromCache(prefix + std::string("up_blocks.2."), weights, memcpyKind);
        up_decoder_block_2d_weight_3->loadWeightsFromCache(prefix + std::string("up_blocks.3."), weights, memcpyKind);
    }
}

template class VaeDecoderWeight<float>;
template class VaeDecoderWeight<half>;
}  // namespace lyradiff