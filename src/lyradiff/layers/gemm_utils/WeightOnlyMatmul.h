#pragma once

#include "GemmProfiler.h"
#include "src/lyradiff/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "src/lyradiff/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
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

constexpr int32_t FP16_BITS       = 16;
constexpr int32_t INT8_BITS       = 8;
constexpr int32_t INT4_BITS       = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;
constexpr int32_t FP16_INT4_RATIO = FP16_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(QuantType weightTypeId)
{
    return weightTypeId == QuantType::INT8_WEIGHT_ONLY ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner    = kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantGemmPluginProfiler:
    public GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig,
                              WeightOnlyGemmRunnerPtr,
                              GemmIdCore,
                              GemmIdCoreHash> {
public:
    using Config = cutlass_extensions::CutlassGemmConfig;

    void setWeightTypeId(QuantType weightId)
    {
        mWeightTypeId = weightId;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    QuantType mWeightTypeId;
};

template<typename DataType>
class WeightOnlyQuantMatmulPlugin {
public:
    using PluginProfilerPtr       = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = delete;

    WeightOnlyQuantMatmulPlugin(const QuantType& weightTypeId, PluginProfilerPtr const& profiler);

    ~WeightOnlyQuantMatmulPlugin(){};

    // IPluginV2DynamicExt Methods
    WeightOnlyQuantMatmulPlugin<DataType>* clone();
    void                                   configurePlugin(int maxM, int K, int N);

    size_t getWorkspaceSize(int M, int K, int N);

    void RunGemm(void const*  A,
                 void const*  B,
                 void const*  weight_scales,
                 void*        C,
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
    void init(const QuantType& weightTypeId);

    void configGemm();

private:
    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t                  m_workspaceMaxSize;

    QuantType mWeightTypeId;
    int       mArch;

    GemmDims   mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

}  // namespace weightonly
}  // namespace lyradiff
