#pragma once
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {
template<typename T>
class ImageProjectBlockWeight {
private:
    size_t image_embed_dim_;
    size_t cross_attention_dim_;
    size_t num_image_text_embeds_;

    bool is_maintain_buffer = false;

protected:
public:
    T* linear_weight = nullptr;
    T* linear_bias   = nullptr;

    T* norm_gamma = nullptr;
    T* norm_beta  = nullptr;

    ~ImageProjectBlockWeight();

    ImageProjectBlockWeight() = default;
    ImageProjectBlockWeight(const size_t image_embed_dim_,
                            const size_t cross_attention_dim_,
                            const size_t num_image_text_embeds);

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};
}  // namespace lyradiff