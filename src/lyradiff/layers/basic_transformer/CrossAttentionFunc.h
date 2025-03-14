#pragma once
#include "src/lyradiff/utils/Tensor.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlockWeight.h"
#include "src/lyradiff/layers/basic_transformer/BasicTransformerBlock.h"

namespace lyradiff {
template<typename T>
using lyra_attn_proc_func = void(TensorMap*, TensorMap*, const BasicTransformerBlockWeight<T>*, BasicTransformerBlock<T>*);
// typedef void(lyra_attn_proc_func) (TensorMap*, TensorMap*, const BasicTransformerBlockWeight<T>*)

template<typename T>
void CrossAttnProcessorBasicFunc(TensorMap* output_tensors, TensorMap* input_tensors, const BasicTransformerBlockWeight<T>* weights, BasicTransformerBlock<T>* basic_transformer_block);
}  // namespace lyradiff