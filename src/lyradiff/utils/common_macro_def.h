#define MACROReMallocWithName(name, block_name, size, is_host)                                                         \
    name = (T*)allocator_->reMallocWithName(block_name "_" #name, size, is_host)

#define MACRODeviceMallocInt8Weights(prefix, n, k)                                                                     \
    deviceMalloc(&prefix##_weight_int8, n* k);                                                                         \
    deviceMalloc(&prefix##_weight_scale, n);                                                                           \
    deviceMalloc(&prefix##_input_quant_scale, 1);                                                                      \
    deviceMalloc(&prefix##_pre_quant_scale, k);

#define MACRODeviceFreeInt8Weights(prefix)                                                                             \
    deviceFree(prefix##_weight_int8);                                                                                  \
    deviceFree(prefix##_weight_scale);                                                                                 \
    deviceFree(prefix##_input_quant_scale);                                                                            \
    deviceFree(prefix##_pre_quant_scale);

#define MACROReMallocWithNameAddOverallSize(name, block_name, size, is_host)                                           \
    name = (T*)allocator_->reMallocWithName(block_name "_" #name, size, is_host);                                      \
    overall_size += size;

#define MACROReMallocWithNameAddOverallSize2(name, block_name, size, is_host)                                          \
    MACROReMallocWithNameAddOverallSize(name, block_name, size, is_host);                                              \
    map_alloced_buff_.emplace(make_pair(#name, name));

#define MACROLoadQKVWeightFromBin(weight_name, attn, offset_sz, ...)                                                   \
    offset = 0;                                                                                                        \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, prefix + #attn ".to_q.weight.bin", model_file_type);       \
    offset += offset_sz;                                                                                               \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, prefix + #attn ".to_k.weight.bin", model_file_type);       \
    offset += offset_sz;                                                                                               \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, prefix + #attn ".to_v.weight.bin", model_file_type);

#define MACROLoadQWeightFromBin(weight_name, attn, offset_sz, ...)                                                     \
    loadWeightFromBin<T>(weight_name, __VA_ARGS__, prefix + #attn ".to_q.weight.bin", model_file_type);

#define MACROLoadKVWeightFromBin(weight_name, attn, offset_sz, ...)                                                    \
    offset = 0;                                                                                                        \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, prefix + #attn ".to_k.weight.bin", model_file_type);       \
    offset += offset_sz;                                                                                               \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, prefix + #attn ".to_v.weight.bin", model_file_type);

#define MACROLoadKVWeightFromBin2(weight_name, fpath_k, fpath_v, offset_sz, ...)                                       \
    offset = 0;                                                                                                        \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, fpath_k, model_file_type);                                 \
    offset += offset_sz;                                                                                               \
    loadWeightFromBin<T>(&weight_name[offset], __VA_ARGS__, fpath_v, model_file_type);

#define MACROLoadLinerarWeightFromBin(weight_name, dim1, dim2, name)                                                   \
    loadWeightFromBin<T>(weight_name##_weight, {dim1, dim2}, prefix + name ".weight.bin", model_file_type);            \
    loadWeightFromBin<T>(weight_name##_bias, {dim1}, prefix + name ".bias.bin", model_file_type);

#define MACROLoadLinearInt8WeightFromBin(weight_name, dim1, dim2, name)                                                \
    loadWeightFromBin<int8_t>(                                                                                         \
        weight_name##_weight_int8, {dim1, dim2}, prefix + name + ".weight_int8.bin", FtCudaDataType::INT8);            \
    loadWeightFromBin<float>(                                                                                          \
        weight_name##_weight_scale, {dim1}, prefix + name + ".weight_quant_scale.bin", FtCudaDataType::FP32);          \
    loadWeightFromBin<float>(                                                                                          \
        weight_name##_input_quant_scale, {1}, prefix + name + ".input_quant_scale.bin", FtCudaDataType::FP32);         \
    loadWeightFromBin<float>(                                                                                          \
        weight_name##_pre_quant_scale, {dim2}, prefix + name + ".pre_quant_scale.bin", FtCudaDataType::FP32);
// loadWeightFromBin<T>(weight_name##_bias, {dim1}, prefix + name ".bias.bin", model_file_type);

#define MACROLoadNormWeightFromBin(weight_name, dim1, name)                                                            \
    loadWeightFromBin<T>(weight_name##_gamma, {dim1}, prefix + name ".weight.bin", model_file_type);                   \
    loadWeightFromBin<T>(weight_name##_beta, {dim1}, prefix + name ".bias.bin", model_file_type);

#define MACROLoadFP8WeightFromBin(weight_name, dim1, dim2, name)                                                       \
    loadWeightFromBin<T>(tmp_buffer, {dim1}, prefix + name ".weight.bin", model_file_type);                            \
    cudaMemsetAsync(weight_name##_weight_scale, 0, sizeof(float), cudaStreamDefault);                                  \
    invokeGetFP8WeightScale(weight_name##_weight_scale, tmp_buffer, dim1, cudaStreamDefault);                          \
    invokeCudaD2DScaleCpyConvert(                                                                                      \
        weight_name##_weight, tmp_buffer, weight_name##_weight_scale, true, dim1, cudaStreamDefault);                  \
    cudaMemcpyAsync(weight_name##_weight_h, tmp_buffer, sizeof(T) * dim1, cudaMemcpyDeviceToHost, cudaStreamDefault);  \
    loadWeightFromBin<T>(weight_name##_bias, {dim2}, prefix + name ".bias.bin", model_file_type);                      \
    loadWeightFromBin<float>(weight_name##_input_scale, {1}, prefix + name "_input_scale.bin", FtCudaDataType::FP32);

#define MACROLoadFP8WeightFromCache(weight_name, dim1, dim2, name)                                                     \
    void* tmp_##weight_name##_weight      = weights[prefix + name ".weight"];                                          \
    void* tmp_##weight_name##_input_scale = weights[prefix + name "_input_scale"];                                     \
    void* tmp_##weight_name##_bias        = weights[prefix + name ".bias"];                                            \
    cudaMemcpyAsync(tmp_buffer, tmp_##weight_name##_weight, sizeof(T) * dim1, memcpyKind, cudaStreamDefault);          \
    cudaMemcpyAsync(weight_name##_bias, tmp_##weight_name##_bias, sizeof(T) * dim2, memcpyKind, cudaStreamDefault);    \
    cudaMemcpyAsync(                                                                                                   \
        weight_name##_input_scale, tmp_##weight_name##_input_scale, sizeof(float), memcpyKind, cudaStreamDefault);     \
    cudaMemsetAsync(weight_name##_weight_scale, 0, sizeof(float), cudaStreamDefault);                                  \
    invokeGetFP8WeightScale(weight_name##_weight_scale, tmp_buffer, dim1, cudaStreamDefault);                          \
    invokeCudaD2DScaleCpyConvert(                                                                                      \
        weight_name##_weight, tmp_buffer, weight_name##_weight_scale, true, dim1, cudaStreamDefault);                  \
    cudaMemcpyAsync(weight_name##_weight_h,                                                                            \
                    tmp_##weight_name##_weight,                                                                        \
                    sizeof(T) * dim1,                                                                                  \
                    cudaMemcpyHostToHost,                                                                              \
                    cudaStreamDefault);

#define MACRODeclareCommonTypes(clz)                                                                                   \
    template class clz<float>;                                                                                         \
    template class clz<half>;

#define MACROFreeBuffByDeviceMalloc(buff)                                                                              \
    delete buff;                                                                                                       \
    buff = nullptr;
