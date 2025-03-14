#pragma once
#include "src/lyradiff/interface/IFLoraWeight.h"
#include "src/lyradiff/layers/timestep_embedding_block/TimestepEmbeddingBlockWeight.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
class TextTimeEmbeddingBlockWeight: public IFLoraWeight<T> {
private:
    size_t timestep_input_dim_;
    size_t augemb_input_dim_;
    size_t output_dim_0_;
    size_t output_dim_;

protected:
    bool is_maintain_buffer = false;

public:
    TimestepEmbeddingBlockWeight<T>* timestep_emb_weight;
    TimestepEmbeddingBlockWeight<T>* augemb_weight;

    TextTimeEmbeddingBlockWeight() = default;
    TextTimeEmbeddingBlockWeight(const size_t        timestep_input_dim,
                                 const size_t        augemb_input_dim,
                                 const size_t        output_dim_0,
                                 const size_t        output_dim,
                                 const LyraQuantType quant_level = LyraQuantType::NONE,
                                 IAllocator*         allocator   = nullptr);

    ~TextTimeEmbeddingBlockWeight();

    virtual void loadWeights(std::string prefix, FtCudaDataType model_file_type);
    virtual void loadWeightsFromCache(std::string                             prefix,
                                      std::unordered_map<std::string, void*>& weights,
                                      cudaMemcpyKind                          memcpyKind);
    virtual void mallocWeights();
};

}  // namespace lyradiff
