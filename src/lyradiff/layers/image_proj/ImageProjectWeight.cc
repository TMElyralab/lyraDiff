#include "src/lyradiff/layers/image_proj/ImageProjectWeight.h"
#include "src/lyradiff/utils/memory_utils.h"

using namespace std;

namespace lyradiff {

template<typename T>
ImageProjectBlockWeight<T>::ImageProjectBlockWeight(const size_t image_embed_dim_,
                                                    const size_t cross_attention_dim_,
                                                    const size_t num_image_text_embeds_):
    image_embed_dim_(image_embed_dim_),
    cross_attention_dim_(cross_attention_dim_),
    num_image_text_embeds_(num_image_text_embeds_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
ImageProjectBlockWeight<T>::~ImageProjectBlockWeight()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_maintain_buffer) {
        deviceFree(linear_weight);
        deviceFree(linear_bias);
        deviceFree(norm_gamma);
        deviceFree(norm_beta);

        linear_weight = nullptr;
        linear_bias   = nullptr;
        norm_gamma    = nullptr;
        norm_beta     = nullptr;
    }
}

template<typename T>
void ImageProjectBlockWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cout << "malloc_weights" << endl;
    deviceMalloc(&linear_weight, image_embed_dim_ * cross_attention_dim_ * num_image_text_embeds_);
    deviceMalloc(&linear_bias, cross_attention_dim_ * num_image_text_embeds_);
    deviceMalloc(&norm_gamma, cross_attention_dim_);
    deviceMalloc(&norm_beta, cross_attention_dim_);

    is_maintain_buffer = true;
}

template<typename T>
void ImageProjectBlockWeight<T>::loadWeights(std::string prefix, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    cout << prefix + ".proj.weight.bin" << endl;
    cout << "image_embed_dim_: " << image_embed_dim_ << endl;
    cout << "cross_attention_dim_: " << cross_attention_dim_ << endl;
    cout << "num_image_text_embeds_: " << num_image_text_embeds_ << endl;
    loadWeightFromBin<T>(linear_weight,
                         {image_embed_dim_, cross_attention_dim_ * num_image_text_embeds_},
                         prefix + ".proj.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        linear_bias, {cross_attention_dim_ * num_image_text_embeds_}, prefix + ".proj.bias.bin", model_file_type);
    loadWeightFromBin<T>(norm_gamma, {cross_attention_dim_}, prefix + ".norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(norm_beta, {cross_attention_dim_}, prefix + ".norm.bias.bin", model_file_type);
}

template<typename T>
void ImageProjectBlockWeight<T>::loadWeightsFromCache(std::string                             prefix,
                                                      std::unordered_map<std::string, void*>& weights,
                                                      cudaMemcpyKind                          memcpyKind)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!is_maintain_buffer) {
        mallocWeights();
    }
    void* tmp_linear_weight = weights[prefix + "linear_1.weight"];
    void* tmp_linear_bias   = weights[prefix + "linear_1.bias"];
    void* tmp_norm_gamma    = weights[prefix + "norm.weight"];
    void* tmp_norm_beta     = weights[prefix + "norm.bias"];

    cudaMemcpy(linear_weight,
               tmp_linear_weight,
               sizeof(T) * image_embed_dim_ * cross_attention_dim_ * num_image_text_embeds_,
               memcpyKind);
    cudaMemcpy(linear_bias, tmp_linear_bias, sizeof(T) * cross_attention_dim_ * num_image_text_embeds_, memcpyKind);
    cudaMemcpy(norm_gamma, tmp_norm_gamma, sizeof(T) * cross_attention_dim_, memcpyKind);
    cudaMemcpy(norm_beta, tmp_norm_beta, sizeof(T) * cross_attention_dim_, memcpyKind);
}

template class ImageProjectBlockWeight<float>;
template class ImageProjectBlockWeight<half>;

}  // namespace lyradiff