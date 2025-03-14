
#include "WeightOnlyMatmul.h"

namespace lyradiff {

namespace gemm {

void WeightOnlyQuantGemmPluginProfiler::runTactic(int                                              m,
                                                  int                                              n,
                                                  int                                              k,
                                                  WeightOnlyQuantGemmPluginProfiler::Config const& tactic,
                                                  char*                                            workspace,
                                                  cudaStream_t const&                              stream)
{
    int const packed_N = int(n / getWeightTypeMultiplier(mWeightTypeId));
    half*     actPtr   = reinterpret_cast<half*>(workspace);
    int8_t*   weightPtr =
        reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* scalesPtr =
        reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), packed_N * k * sizeof(int8_t)));
    half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scalesPtr), n * sizeof(half)));
    char* workspacePtr =
        reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * n * sizeof(half)));

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    if (mWeightTypeId == QuantType::INT8_WEIGHT_ONLY) {
        mRunner->gemm(actPtr, weightPtr, scalesPtr, outputPtr, m, n, k, tactic, workspacePtr, wsSize, stream);
    }
    else {
        mRunner->gemm(actPtr,
                      reinterpret_cast<cutlass::uint4b_t*>(weightPtr),
                      scalesPtr,
                      outputPtr,
                      m,
                      n,
                      k,
                      tactic,
                      workspacePtr,
                      wsSize,
                      stream);
    }
}

void WeightOnlyQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    int const           packed_N   = int(n / getWeightTypeMultiplier(mWeightTypeId));
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),               // A
        packed_N * k * sizeof(int8_t),         // B
        n * sizeof(half),                      // scales
        maxM * n * sizeof(half),               // C
        mRunner->getWorkspaceSize(maxM, n, k)  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyQuantGemmPluginProfiler::Config>
WeightOnlyQuantGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

template<typename DataType>
WeightOnlyQuantMatmulPlugin<DataType>::WeightOnlyQuantMatmulPlugin(
    const QuantType& weightTypeId, WeightOnlyQuantMatmulPlugin<DataType>::PluginProfilerPtr const& pluginProfiler):
    mPluginProfiler(pluginProfiler)
{
    init(weightTypeId);
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::init(const QuantType& weightTypeId)
{
    mArch         = getSMVersion();
    mWeightTypeId = weightTypeId;

    if (mWeightTypeId == QuantType::INT8_WEIGHT_ONLY) {
        m_weightOnlyGemmRunner = std::make_shared<
            kernels::cutlass_kernels::
                CutlassFpAIntBGemmRunner<DataType, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
    }
    else if (mWeightTypeId == QuantType::PACKED_INT4_WEIGHT_ONLY) {

        m_weightOnlyGemmRunner = std::make_shared<
            kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<DataType,
                                                               cutlass::uint4b_t,
                                                               cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
    }
    else {
        LYRA_CHECK(false);
    }

    mPluginProfiler->setWeightTypeId(mWeightTypeId);

    mGemmId = GemmIdCore(mDims.n, mDims.k);
}

// IPluginV2DynamicExt Methods
template<typename DataType>
WeightOnlyQuantMatmulPlugin<DataType>* WeightOnlyQuantMatmulPlugin<DataType>::clone()
{
    auto* plugin = new WeightOnlyQuantMatmulPlugin(*this);
    return plugin;
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGemmRunner, mDims, mGemmId);
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::configurePlugin(int maxM, int K, int N)
{
    mDims = {1, maxM, N, K};

    mGemmId = {N, K};

    m_workspaceMaxSize = m_weightOnlyGemmRunner->getWorkspaceSize(maxM, N, K);
}

template<typename DataType>
size_t WeightOnlyQuantMatmulPlugin<DataType>::getWorkspaceSize(int M, int K, int N)
{
    return m_weightOnlyGemmRunner->getWorkspaceSize(M, N, K);
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::RunGemm(void const*  A,
                                                    void const*  B,
                                                    void const*  weight_scales,
                                                    void*        C,
                                                    int          m,
                                                    int          n,
                                                    int          k,
                                                    void*        workspace,
                                                    cudaStream_t stream)
{
#ifdef ENABLE_BF16
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value || std::is_same<DataType, __nv_bfloat16>::value),
                         "No valid weightOnlyQuantMatmul configuration");
#else
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value), "No valid weightOnlyQuantMatmul configuration");
#endif

    int const ws_size = m_weightOnlyGemmRunner->getWorkspaceSize(m, n, k);

    GemmIdCore cur_gemm_id = {n, k};

    auto const& bestTactic = mPluginProfiler->getBestConfig(m, cur_gemm_id);
    LYRA_CHECK_WITH_INFO(
        bestTactic,
        "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    m_weightOnlyGemmRunner->gemm(
        A, B, weight_scales, C, m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), ws_size, stream);
}

template<typename DataType>
int WeightOnlyQuantMatmulPlugin<DataType>::initialize()
{
    configGemm();
    return 0;
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::terminate()
{
}

template<typename DataType>
size_t WeightOnlyQuantMatmulPlugin<DataType>::getSerializationSize()
{
    return sizeof(mWeightTypeId) +                          // mWeightTypeId
           sizeof(mDims) +                                  // Dimensions
           mPluginProfiler->getSerializationSize(mGemmId);  // selected tactics container size
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::serialize(void* buffer)
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mWeightTypeId);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

template<typename DataType>
void WeightOnlyQuantMatmulPlugin<DataType>::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

template class WeightOnlyQuantMatmulPlugin<half>;
#ifdef ENABLE_BF16
template class WeightOnlyQuantMatmulPlugin<__nv_bfloat16>;
#endif

}  // namespace gemm
}  // namespace lyradiff
