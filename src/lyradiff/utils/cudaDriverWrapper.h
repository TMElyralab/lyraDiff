
#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include <cstdio>
#include <cuda.h>

#ifndef PLUGIN_ASSERT
#define PLUGIN_ASSERT(cond)                                                                                            \
    {                                                                                                                  \
        if ((cond) == false) {                                                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    }
#endif

#define cuErrCheck(stat, wrap)                                                                                         \
    {                                                                                                                  \
        cuErrCheck_((stat), wrap, __FILE__, __LINE__);                                                                 \
    }

namespace lyradiff {

class CUDADriverWrapper {
public:
    CUDADriverWrapper();

    ~CUDADriverWrapper();

    CUresult cuGetErrorName(CUresult error, const char** pStr) const;

    CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const;

    CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

    CUresult cuModuleUnload(CUmodule hmod) const;

    CUresult cuLinkDestroy(CUlinkState state) const;

    CUresult cuModuleLoadData(CUmodule* module, const void* image) const;

    CUresult
    cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) const;

    CUresult cuLinkAddFile(CUlinkState    state,
                           CUjitInputType type,
                           const char*    path,
                           unsigned int   numOptions,
                           CUjit_option*  options,
                           void**         optionValues) const;

    CUresult cuLinkAddData(CUlinkState    state,
                           CUjitInputType type,
                           void*          data,
                           size_t         size,
                           const char*    name,
                           unsigned int   numOptions,
                           CUjit_option*  options,
                           void**         optionValues) const;

    CUresult cuLaunchCooperativeKernel(CUfunction   f,
                                       unsigned int gridDimX,
                                       unsigned int gridDimY,
                                       unsigned int gridDimZ,
                                       unsigned int blockDimX,
                                       unsigned int blockDimY,
                                       unsigned int blockDimZ,
                                       unsigned int sharedMemBytes,
                                       CUstream     hStream,
                                       void**       kernelParams) const;

    CUresult cuLaunchKernel(CUfunction   f,
                            unsigned int gridDimX,
                            unsigned int gridDimY,
                            unsigned int gridDimZ,
                            unsigned int blockDimX,
                            unsigned int blockDimY,
                            unsigned int blockDimZ,
                            unsigned int sharedMemBytes,
                            CUstream     hStream,
                            void**       kernelParams,
                            void**       extra) const;

private:
    void* handle;
    CUresult (*_cuGetErrorName)(CUresult, const char**);
    CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
    CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
    CUresult (*_cuModuleUnload)(CUmodule);
    CUresult (*_cuLinkDestroy)(CUlinkState);
    CUresult (*_cuLinkCreate)(unsigned int, CUjit_option*, void**, CUlinkState*);
    CUresult (*_cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLinkAddData)(
        CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLaunchCooperativeKernel)(CUfunction,
                                           unsigned int,
                                           unsigned int,
                                           unsigned int,
                                           unsigned int,
                                           unsigned int,
                                           unsigned int,
                                           unsigned int,
                                           CUstream,
                                           void**);
    CUresult (*_cuLaunchKernel)(CUfunction   f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream     hStream,
                                void**       kernelParams,
                                void**       extra);
};

inline void cuErrCheck_(CUresult stat, const CUDADriverWrapper& wrap, const char* file, int line)
{
    if (stat != CUDA_SUCCESS) {
        const char* msg = nullptr;
        wrap.cuGetErrorName(stat, &msg);
        fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
    }
}

}  // namespace lyradiff

#endif  // CUDA_DRIVER_WRAPPER_H
