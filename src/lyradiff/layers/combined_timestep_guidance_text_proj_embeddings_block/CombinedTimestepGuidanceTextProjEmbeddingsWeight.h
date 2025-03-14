#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class CombinedTimestepGuidanceTextProjEmbeddingsWeight: public IFLoraWeight<T> {
private:
    size_t pooled_projection_dim_;
    size_t embedding_dim_;
    size_t embedding_input_dim_ = 256;  // 写死在FLUX当中

protected:
    bool is_maintain_buffer = false;

public:
    TimestepEmbeddingBlockWeight<T>* timestep_emb_weight;
    TimestepEmbeddingBlockWeight<T>* guidance_emb_weight;
    TimestepEmbeddingBlockWeight<T>* text_emb_weight;

    CombinedTimestepGuidanceTextProjEmbeddingsWeight() = default;
    CombinedTimestepGuidanceTextProjEmbeddingsWeight(const size_t pooled_projection_dim,
                                                     const size_t embedding_dim,
                                                     const size_t embedding_input_dim = 256,  // 写死在FLUX当中
                                                     const LyraQuantType quant_level  = LyraQuantType::NONE,
                                                     IAllocator*         allocator    = nullptr);

    ~CombinedTimestepGuidanceTextProjEmbeddingsWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
