#pragma once
#include "cuda_runtime.h"
#include "src/lyradiff/layers/BaseLayer.h"
#include "src/lyradiff/layers/image_proj/ImageProjectWeight.h"
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace lyradiff {
template<typename T>
class ImageProjection: public BaseLayer {
private:
    size_t image_embed_dim_;
    size_t cross_attention_dim_;
    size_t num_image_text_embeds_;

    size_t pre_batch_size_ = 0;
    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size);

public:
    T* inter_buf_0_0_ = nullptr;  // for Linear0
    T* inter_buf_0_1_ = nullptr;  // for Linear0 add bias

    ImageProjection(const size_t     image_embed_dim_,
                    const size_t     cross_attention_dim_,
                    const size_t     num_image_text_embeds,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    const bool       is_free_buffer_after_forward,
                    const bool       sparse);

    ImageProjection(ImageProjection const& ImageProjection);

    ~ImageProjection();

    virtual void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const ImageProjectBlockWeight<T>* image_proj_weight);
    virtual void
    forward(Tensor& output_tensors, Tensor& input_tensors, const ImageProjectBlockWeight<T>* image_proj_weight);
};

}  // namespace lyradiff