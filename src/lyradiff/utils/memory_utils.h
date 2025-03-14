/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_fp8_utils.h"
#include "src/lyradiff/utils/cuda_utils.h"

namespace lyradiff {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize = false);

template<typename T>
void deviceMemSetZero(T* ptr, size_t size);

template<typename T>
void deviceFree(T*& ptr);

template<typename T>
void deviceFill(T* devptr, size_t size, T value, cudaStream_t stream = 0);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaRandomUniform(T* buffer, const size_t size);

template<typename T>
int loadWeightFromBin(T*                  ptr,
                      std::vector<size_t> shape,
                      std::string         filename,
                      FtCudaDataType      model_file_type = FtCudaDataType::FP32);

template<typename T>
void invokeInPlaceTranspose(T* data, T* workspace, const int dim0, const int dim1);

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const size_t size, cudaStream_t stream);
#ifdef ENABLE_FP8
void invokeCudaD2Dcpyfp82Float(float* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream);
void invokeCudaD2Dcpyfp82Half(half* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2fp8(__nv_fp8_e4m3* dst, float* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyHalf2fp8(__nv_fp8_e4m3* dst, half* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyBfloat2fp8(__nv_fp8_e4m3* dst, __nv_bfloat16* src, const size_t size, cudaStream_t stream);
#endif  // ENABLE_FP8
#ifdef ENABLE_BF16
void invokeCudaD2DcpyBfloat2Float(float* dst, __nv_bfloat16* src, const size_t size, cudaStream_t stream);
#endif  // ENABLE_BF16

template<typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const size_t size, cudaStream_t stream = 0);

template<typename T_IN, typename T_OUT>
void invokeCudaD2DScaleCpyConvert(
    T_OUT* tgt, const T_IN* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream = 0);

template<typename T>
void invokeMultiplyScale(T* tensor, float scale, const size_t size, cudaStream_t stream);

template<typename T>
void invokeDivideScale(T* tensor, float scale, const size_t size, cudaStream_t stream);

template<typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, const size_t size, cudaStream_t stream = 0);

inline bool checkIfFileExist(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        in.close();
        return true;
    }
    return false;
}

template<typename T>
void saveToBinary(const T* ptr, const size_t size, std::string filename);

template<typename T>
void invokeCalculateSum(const T* data, float* sum, const size_t size);

template<typename T>
void transpose(T* dst, const T* src, const size_t N, const size_t M);

}  // namespace lyradiff
