#pragma once

#include "GemmProfiler.h"
#include "src/lyradiff/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "src/lyradiff/kernels/cutlass_kernels/gemm/gemm.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/workspace.h"

#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace lyradiff {

namespace gemm {

constexpr int32_t FP16_BITS = 16;

using GemmRunner    = kernels::cutlass_kernels::CutlassGemmRunnerInterface;
using GemmRunnerPtr = std::shared_ptr<GemmRunner>;

class GeneralGemmPluginProfiler:
    public GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig, GemmRunnerPtr, GemmIdCore, GemmIdCoreHash> {
public:
    using Config = cutlass_extensions::CutlassGemmConfig;

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;
};

template<typename DataType>
class MatmulPlugin {
public:
    using PluginProfilerPtr = std::shared_ptr<GeneralGemmPluginProfiler>;
    MatmulPlugin()          = delete;
    MatmulPlugin(PluginProfilerPtr const& profiler);

    ~MatmulPlugin(){};

    // IPluginV2DynamicExt Methods
    MatmulPlugin<DataType>* clone();
    void                    configurePlugin(int maxM, int K, int N);

    size_t getWorkspaceSize(int M, int K, int N);

    void RunGemm(void const*  A,
                 void const*  B,
                 void*        C,
                 void const*  bias,
                 int          m,
                 int          n,
                 int          k,
                 void*        workspace,
                 cudaStream_t stream);

    // IPluginV2 Methods
    int    initialize();
    void   terminate();
    size_t getSerializationSize();
    void   serialize(void* buffer);
    void   destroy();

private:
    void init();

    void configGemm();

private:
    GemmRunnerPtr m_GemmRunner;
    size_t        m_workspaceMaxSize;

    int mArch;

    GemmDims   mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

}  // namespace gemm
}  // namespace lyradiff
