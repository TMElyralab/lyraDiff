#include "TimeProjection.h"
#include "src/lyradiff/kernels/time_proj/time_proj_kernels.h"
#include <cmath>

namespace lyradiff {

template<typename T>
TimeProjection<T>::TimeProjection(const size_t     num_channels,
                                  const bool       flip_sin_to_cos,
                                  const size_t     downscale_freq_shift,
                                  const size_t     max_period,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator,
                                  const bool       is_free_buffer_after_forward,
                                  const bool       sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    num_channels_(num_channels),
    flip_sin_to_cos_(flip_sin_to_cos),
    downscale_freq_shift_(downscale_freq_shift),
    max_period_(max_period)
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

    // pre-calculate exponents_
    half_dim_ = num_channels_ / 2;
    exponent_factor_ =
        static_cast<float>(-std::log(static_cast<float>(max_period_)) / (half_dim_ - downscale_freq_shift_));
    float* exponents_temp_ = (float*)malloc(sizeof(float) * half_dim_);

    for (int i = 0; i < half_dim_; i++) {
        exponents_temp_[i] = expf(exponent_factor_ * i);
    }

    size_t exponents_size_ = sizeof(float) * half_dim_;
    exponents_             = (float*)allocator_->reMalloc(exponents_, exponents_size_, false);
    cudaMemcpy(exponents_, exponents_temp_, exponents_size_, cudaMemcpyHostToDevice);

    free(exponents_temp_);
}

template<typename T>
TimeProjection<T>::TimeProjection(const size_t     num_channels,
                                  const bool       flip_sin_to_cos,
                                  const size_t     downscale_freq_shift,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator,
                                  const bool       is_free_buffer_after_forward,
                                  const bool       sparse):
    TimeProjection(num_channels,
                   flip_sin_to_cos,
                   downscale_freq_shift,
                   10000,
                   stream,
                   cublas_wrapper,
                   allocator,
                   is_free_buffer_after_forward,
                   sparse)
{
}

template<typename T>
TimeProjection<T>::TimeProjection(TimeProjection<T> const& time_proj):
    BaseLayer(time_proj.stream_,
              time_proj.cublas_wrapper_,
              time_proj.allocator_,
              time_proj.is_free_buffer_after_forward_,
              time_proj.cuda_device_prop_,
              time_proj.sparse_),
    num_channels_(time_proj.num_channels_),
    flip_sin_to_cos_(time_proj.flip_sin_to_cos_),
    downscale_freq_shift_(time_proj.downscale_freq_shift_),
    max_period_(time_proj.max_period_),
    exponent_factor_(time_proj.exponent_factor_),
    exponents_(time_proj.exponents_)
{
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }
    else {
        throw "Unsupported data type";
    }
}

template<typename T>
void TimeProjection<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    timestep_buffer_    = (float*)allocator_->reMallocWithName("TimeProjection_timestep_buffer_", sizeof(float), false);
    is_allocate_buffer_ = false;
}

template<typename T>
void TimeProjection<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        // allocator_->free((void**)(&timestep_buffer_));
        timestep_buffer_ = nullptr;

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void TimeProjection<T>::forward(Tensor& output_tensor, const float& timestep)
{
    // input tensors:
    //      timestep (float)

    // output tensors:
    //      output: [bs, num_channels]
    T* output_d_ptr = output_tensor.getPtr<T>();

    size_t batch_size = output_tensor.shape[0];

    allocateBuffer();

    cudaMemcpy(timestep_buffer_, &timestep, sizeof(float), cudaMemcpyHostToDevice);

    invokeTimeProjection<T>(
        output_d_ptr, timestep_buffer_, exponents_, flip_sin_to_cos_, half_dim_, batch_size, stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void TimeProjection<T>::forward(TensorMap* output_tensors, const float& timestep)
{
    // input tensors:
    //      timestep (float)

    // output tensors:
    //      output: [bs, num_channels]
    Tensor output_tensor = output_tensors->at("output");
    forward(output_tensor, timestep);
}

template<typename T>
void TimeProjection<T>::forward(Tensor& output_tensor, Tensor& input_tensor)
{

    T* output_d_ptr = output_tensor.getPtr<T>();

    size_t n_steps    = input_tensor.shape[1];
    size_t batch_size = output_tensor.shape[0];
    invokeTimeProjectionMulti<T>(output_d_ptr,
                                 input_tensor.getPtr<float>(),
                                 n_steps,
                                 exponents_,
                                 flip_sin_to_cos_,
                                 half_dim_,
                                 batch_size,
                                 stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template<typename T>
void TimeProjection<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // input tensors:
    //      timestep (float)

    // output tensors:
    //      output: [bs, num_channels]
    Tensor output_tensor = output_tensors->at("output");
    Tensor input_tensor  = input_tensors->at("input");
    forward(output_tensor, input_tensor);
}

template<typename T>
TimeProjection<T>::~TimeProjection()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void**)(&exponents_));
    cublas_wrapper_ = nullptr;
    exponents_      = nullptr;
    // freeBuffer();
}

template class TimeProjection<float>;
template class TimeProjection<half>;
#ifdef ENABLE_BF16
template class TimeProjection<__nv_bfloat16>;
#endif
}  // namespace lyradiff