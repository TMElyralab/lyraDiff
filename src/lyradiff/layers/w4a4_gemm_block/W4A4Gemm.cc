#include "W4A4Gemm.h"
#include "src/lyradiff/kernels/w4a4_gemm/w4a4_gemm.h"
#include "src/lyradiff/utils/cuda_utils.h"
using namespace std;
namespace lyradiff {

template<typename T>
W4A4Gemm<T>::W4A4Gemm(size_t              d_out,
                      size_t              d_in,
                      size_t              lora_rank,
                      bool                has_bias,
                      size_t              group_size,
                      cudaStream_t        stream,
                      cublasMMWrapper*    cublas_wrapper,
                      IAllocator*         allocator,
                      const bool          is_free_buffer_after_forward,
                      const bool          sparse,
                      const LyraQuantType quant_level):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse, quant_level),
    d_out_(d_out),
    d_in_(d_in),
    lora_rank_(lora_rank),
    has_bias_(has_bias),
    group_size_(group_size)
{
    if (std::is_same<T, half>::value) {
        printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to bf16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }

    group_num_ = d_in_ / group_size_;
}

template<typename T>
W4A4Gemm<T>::W4A4Gemm(W4A4Gemm<T> const& other):
    BaseLayer(other.stream_,
              other.cublas_wrapper_,
              other.allocator_,
              other.is_free_buffer_after_forward_,
              other.cuda_device_prop_,
              other.sparse_,
              other.quant_level_),
    d_out_(other.d_out_),
    d_in_(other.d_in_),
    lora_rank_(other.lora_rank_),
    has_bias_(other.has_bias_),
    group_size_(other.group_size_),
    group_num_(other.group_num_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (typeid(T) == typeid(__nv_bfloat16)) {
        // printf("set cublas_wrapper to fp16 mode\n");
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
void W4A4Gemm<T>::allocateBuffer()
{
    LYRA_CHECK_WITH_INFO(
        false, "W4A4Gemm::allocateBuffer() is deprecated. Use `allocateBuffer(size_t batch_size, ...)` instead");
}

template<typename T>
void W4A4Gemm<T>::allocateBuffer(size_t m)
{
    size_t lora_down_size       = sizeof(float) * m * lora_rank_;
    size_t lora_down_size2      = sizeof(T) * m * lora_rank_;
    size_t input_scale_size     = sizeof(float) * m * group_num_;
    size_t quantized_input_size = sizeof(uint32_t) * m * d_in_ / 8;  // 此处写死/8 因为这里是写死 pack int4 到 int32

    lora_down_res_buffer = (float*)allocator_->reMallocWithName(
        "W4A4Gemm_lora_down_res_buffer", lora_down_size, false);  // 这里给 true，因为需要重置 lora down 结果的数值
    cudaMemsetAsync(lora_down_res_buffer, 0, lora_down_size, stream_);
    lora_down_res_buffer2 = (T*)allocator_->reMallocWithName(
        "W4A4Gemm_lora_down_res_buffer2", lora_down_size2, false);  // 这里给 true，因为需要重置 lora down 结果的数值
    input_scale_buffer =
        (float*)allocator_->reMallocWithName("W4A4Gemm_context_input_scale_buffer", input_scale_size, false);
    quantized_input_buffer =
        (uint32_t*)allocator_->reMallocWithName("W4A4Gemm_quantized_input_buffer", quantized_input_size, false);
    // msa_buffer  = norm_buffer2;
}

template<typename T>
void W4A4Gemm<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void W4A4Gemm<T>::forward(const TensorMap*         output_tensors,
                          const TensorMap*         input_tensors,
                          const W4A4GemmWeight<T>* weights)
{
    Tensor input_tensor = input_tensors->at("input");

    Tensor output_tensor = output_tensors->at("output");

    size_t m = input_tensor.shape[0];
    size_t k = input_tensor.shape[1];
    size_t n = output_tensor.shape[1];

    assert(k == d_in_);
    assert(n == d_out_);

    // cout << "dout: " << d_out_ << endl;
    // cout << "d_in_: " << d_in_ << endl;
    // cout << "m: " << m << endl;
    // cout << "n: " << n << endl;
    // cout << "k: " << k << endl;
    // cout << "group_num_: " << group_num_ << endl;

    allocateBuffer(m);

    // cudaMemset(&stage2_res, 0, sizeof(__nv_bfloat16) * stage2_m * stage2_n);
    invokeFusedQuantizeAndLoraDownSimple(lora_down_res_buffer,
                                         quantized_input_buffer,
                                         input_scale_buffer,
                                         input_tensor.getPtr<T>(),
                                         weights->lora_down,
                                         weights->smooth,
                                         m,
                                         lora_rank_,
                                         k,
                                         group_num_,
                                         stream_);

    invokeCudaD2DcpyConvert(lora_down_res_buffer2, lora_down_res_buffer, m * lora_rank_);

    invokeFusedW4A4GemmAndLoraUp(output_tensor.getPtr<T>(),
                                 quantized_input_buffer,
                                 weights->packed_weight,
                                 lora_down_res_buffer2,
                                 weights->lora_up,
                                 input_scale_buffer,
                                 weights->weight_scale,
                                 m,
                                 n,
                                 k,
                                 lora_rank_,
                                 group_num_,
                                 stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
W4A4Gemm<T>::~W4A4Gemm()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class W4A4Gemm<float>;
template class W4A4Gemm<half>;
#ifdef ENABLE_BF16
template class W4A4Gemm<__nv_bfloat16>;
#endif
}  // namespace lyradiff