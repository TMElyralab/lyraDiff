#pragma once

#include "commonDatatype.h"
#include "src/lyradiff/utils/cudaDriverWrapper.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace lyradiff {
namespace flash_attn {

template<typename TKernelMeta, typename TKernelParam>
class TSharedCubinKernel {
public:
    using KernelMeta  = TKernelMeta;
    using KernelParam = TKernelParam;

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const = 0;
    virtual uint64_t hashID(TKernelParam const& param) const    = 0;

    TSharedCubinKernel(TKernelMeta const* pMetaStart, int32_t nMetaCount, MHFADataType type, int32_t sm):
        mDataType(type), mKernelMeta(pMetaStart), mKernelMetaCount(nMetaCount), mSM(sm)
    {
        PLUGIN_ASSERT(mKernelMetaCount && "No kernels were loaded correctly.");
    }

    void loadCubinKernels(int32_t smVersion)
    {
        for (int32_t i = 0; i < mKernelMetaCount; ++i) {
            auto const& kernelMeta = mKernelMeta[i];
            auto const  kernelKey  = hashID(kernelMeta);
            if (kernelMeta.mSM == smVersion && kernelMeta.mDataType == mDataType
                && mFunctions.find(kernelKey) == mFunctions.end()) {
                int32_t const DEFAULT_SMEM_SIZE{48 * 1024};
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE) {
                    int32_t deviceID{0};
                    cudaGetDevice(&deviceID);
                    int32_t sharedMemPerMultiprocessor{0};
                    if (cudaDeviceGetAttribute(
                            &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID)
                            != cudaSuccess
                        || sharedMemPerMultiprocessor < kernelMeta.mSharedMemBytes) {
                        // skip load function because not enough shared memory to launch the kernel
                        continue;
                    }
                }

                CUmodule hmod{0};
                auto     findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end()) {
                    hmod = findModuleIter->second;
                }
                else {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE) {
                    if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                   kernelMeta.mSharedMemBytes)
                        != CUDA_SUCCESS) {
                        // some chip may not have enough shared memory to launch the kernel
                        continue;
                    }
                }
                mFunctions.insert({kernelKey, funcInfo});
            }
        }
    }

    void loadCubinKernels()
    {
        if (!mFunctions.empty()) {
            return;
        }

        loadCubinKernels(mSM);
    }

    bool isValid(int32_t s) const
    {
        return !mFunctions.empty();
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        if (params.interleaved) {
            PLUGIN_ASSERT(mDataType == DATA_TYPE_INT8);
        }
        auto const findIter = mFunctions.find(hashID(params));
        PLUGIN_ASSERT(findIter != mFunctions.end());

        auto const&      kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        CUfunction const func       = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        if (!params.force_unroll) {
            cuErrCheck(mDriver.cuLaunchKernel(func,
                                              params.h,
                                              params.b,
                                              1,
                                              kernelMeta.mThreadsPerCTA,
                                              1,
                                              1,
                                              kernelMeta.mSharedMemBytes,
                                              ss,
                                              kernelParams,
                                              nullptr),
                       mDriver);
        }
        else {
            int32_t unroll = (params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep;
            cuErrCheck(mDriver.cuLaunchKernel(func,
                                              params.h,
                                              params.b,
                                              unroll,
                                              kernelMeta.mThreadsPerCTA,
                                              1,
                                              1,
                                              kernelMeta.mSharedMemBytes,
                                              ss,
                                              kernelParams,
                                              nullptr),
                       mDriver);
        }
    }

    virtual ~TSharedCubinKernel() = default;

protected:
    lyradiff::CUDADriverWrapper mDriver;

    MHFADataType                                       mDataType;
    TKernelMeta const*                                 mKernelMeta;
    int32_t                                            mKernelMetaCount;
    int32_t                                            mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo {
        int32_t    mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
};

template<typename TKernelList>
class TSharedCubinKernelFactory {
public:
    TKernelList const* getCubinKernels(typename TKernelList::KernelMeta const* pKernelList,
                                       int32_t                                 nbKernels,
                                       MHFADataType                            type,
                                       int32_t                                 sm)
    {
        static std::mutex           sMutex;
        std::lock_guard<std::mutex> lg(sMutex);

        auto const id       = hashID(type, sm);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end()) {
            auto* newKernel = new TKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadCubinKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TSharedCubinKernelFactory<TKernelList>& Get()
    {
        static TSharedCubinKernelFactory<TKernelList> gFactory;
        return gFactory;
    }

private:
    TSharedCubinKernelFactory() = default;

    inline uint64_t hashID(MHFADataType type, int32_t sm) const
    {
        // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
        int32_t deviceID{0};
        cudaGetDevice(&deviceID);

        PLUGIN_ASSERT((deviceID & 0xFFFF) == deviceID);
        PLUGIN_ASSERT((type & 0xFFFF) == type);
        return (uint64_t)type << 48 | (uint64_t)deviceID << 32 | sm;
    }

    std::unordered_map<uint64_t, std::unique_ptr<TKernelList> const> mKernels;
};

}  // namespace flash_attn
}  // namespace lyradiff
