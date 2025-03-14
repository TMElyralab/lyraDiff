#pragma once

#include "GemmProfiler.h"
#include "src/lyradiff/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "src/lyradiff/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/quantization.h"
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

// using perfMapType     = std::unordered_map<int, cutlass_extensions::CutlassGemmConfig>;
using SqGemmRunnerPtr = std::shared_ptr<kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>;

class SmoothQuantGemmPluginProfiler:
    public GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig, SqGemmRunnerPtr, GemmIdCore, GemmIdCoreHash> {
public:
    using Config = cutlass_extensions::CutlassGemmConfig;

    void setQuantMode(QuantMode const& quantMode)
    {
        mQuantMode = quantMode;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    QuantMode mQuantMode;
};

template<typename DataType>
class SmoothQuantGemmPlugin {
public:
    using PluginProfilerPtr = std::shared_ptr<SmoothQuantGemmPluginProfiler>;
    SmoothQuantGemmPlugin() = delete;

    SmoothQuantGemmPlugin(const QuantMode& quantMode, PluginProfilerPtr const& profiler);

    SmoothQuantGemmPlugin(const SmoothQuantGemmPlugin& plugin);

    ~SmoothQuantGemmPlugin(){};

    // IPluginV2DynamicExt Methods
    SmoothQuantGemmPlugin<DataType>* clone();
    void                             configurePlugin(int maxM, int K, int N);

    size_t getWorkspaceSize(int M, int K, int N);

    void RunGemm(int8_t const* A,
                 int8_t const* B,
                 float const*  alphaCol,
                 float const*  alphaRow,
                 void*         C,
                 int           m,
                 int           n,
                 int           k,
                 char*         workspacePtr,
                 cudaStream_t  stream);

    // IPluginV2 Methods
    int    initialize();
    void   terminate();
    size_t getSerializationSize();
    void   serialize(void* buffer);
    void   destroy();

    QuantMode mQuantMode;

private:
    void init();

    void configGemm();

protected:
    SqGemmRunnerPtr m_sqGemmRunner;
    size_t          m_workspaceMaxSize;

    int mArch;

    GemmDims   mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

}  // namespace gemm
}  // namespace lyradiff
