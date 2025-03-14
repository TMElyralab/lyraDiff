#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lyradiff {

template<typename T_OUT, typename T_IN>
void invokeTensorD2DConvert(T_OUT* tgt, const T_IN* src, const size_t size, cudaStream_t stream);
}
