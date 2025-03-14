

#pragma once

// #include "src/lyradiff/utils/quantization.h"
#include "src/lyradiff/kernels/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>

namespace tk  = lyradiff;
namespace tkc = lyradiff::cutlass_extensions;

namespace lyradiff {
namespace kernels {
namespace cutlass_kernels {

class CutlassGemmRunnerInterface {
public:
    CutlassGemmRunnerInterface() {}

    virtual ~CutlassGemmRunnerInterface() {}

    virtual void gemm(void const*            A,
                      void const*            B,
                      void*                  C,
                      void const*            bias,
                      int                    m,
                      int                    n,
                      int                    k,
                      tkc::CutlassGemmConfig gemmConfig,
                      char*                  workspace,
                      size_t                 workspaceBytes,
                      cudaStream_t           stream) = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE    = 32;
    static constexpr int MIN_N_TILE    = 64;
};

template<typename T>
class CutlassGemmRunner: public virtual CutlassGemmRunnerInterface {
public:
    CutlassGemmRunner();
    ~CutlassGemmRunner();

    void gemm(void const*            A,
              void const*            B,
              void*                  C,
              void const*            bias,
              int                    m,
              int                    n,
              int                    k,
              tkc::CutlassGemmConfig gemmConfig,
              char*                  workspace,
              size_t                 workspaceBytes,
              cudaStream_t           stream) override;

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(int const m, int const n, int const k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    void dispatchToArch(void const*            A,
                        void const*            B,
                        void*                  C,
                        void const*            bias,
                        int                    m,
                        int                    n,
                        int                    k,
                        tkc::CutlassGemmConfig gemmConfig,
                        char*                  workspace,
                        size_t                 workspaceBytes,
                        cudaStream_t           stream);

    int mSm;
    int mMultiProcessorCount;
};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace lyradiff
