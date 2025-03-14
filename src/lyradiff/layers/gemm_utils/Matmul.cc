
#include "Matmul.h"

namespace lyradiff {

namespace gemm {

void GeneralGemmPluginProfiler::runTactic(
    int m, int n, int k, GeneralGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    half* input     = reinterpret_cast<half*>(workspace);
    half* weightPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(input), m * k * sizeof(half)));
    half* outputPtr =
        reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(half)));
    char* workspacePtr =
        reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * n * sizeof(half)));

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(input, weightPtr, outputPtr, nullptr, m, n, k, tactic, workspacePtr, wsSize, stream);
}

void GeneralGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),               // A
        n * k * sizeof(half),                  // B
        maxM * n * sizeof(half),               // C
        mRunner->getWorkspaceSize(maxM, n, k)  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<GeneralGemmPluginProfiler::Config> GeneralGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

template<typename DataType>
MatmulPlugin<DataType>::MatmulPlugin(MatmulPlugin<DataType>::PluginProfilerPtr const& pluginProfiler):
    mPluginProfiler(pluginProfiler)
{
    init();
}

template<typename DataType>
void MatmulPlugin<DataType>::init()
{
    mArch = getSMVersion();

    m_GemmRunner = std::make_shared<kernels::cutlass_kernels::CutlassGemmRunner<DataType>>();

    mGemmId = GemmIdCore(mDims.n, mDims.k);
}

// IPluginV2DynamicExt Methods
template<typename DataType>
MatmulPlugin<DataType>* MatmulPlugin<DataType>::clone()
{
    auto* plugin = new MatmulPlugin(*this);
    return plugin;
}

template<typename DataType>
void MatmulPlugin<DataType>::configGemm()
{
    mPluginProfiler->profileTactics(m_GemmRunner, mDims, mGemmId);
}

template<typename DataType>
void MatmulPlugin<DataType>::configurePlugin(int maxM, int K, int N)
{
    if (!mDims.isInitialized()) {
        mDims = {1, maxM, N, K};
    }

    mGemmId = {N, K};

    m_workspaceMaxSize = m_GemmRunner->getWorkspaceSize(maxM, N, K);
}

template<typename DataType>
size_t MatmulPlugin<DataType>::getWorkspaceSize(int M, int K, int N)
{
    return m_GemmRunner->getWorkspaceSize(M, N, K);
}

template<typename DataType>
void MatmulPlugin<DataType>::RunGemm(
    void const* A, void const* B, void* C, void const* bias, int m, int n, int k, void* workspace, cudaStream_t stream)
{
#ifdef ENABLE_BF16
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value || std::is_same<DataType, __nv_bfloat16>::value),
                         "No valid Matmul configuration");
#else
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value), "No valid Matmul configuration");
#endif

    int const ws_size = m_GemmRunner->getWorkspaceSize(m, n, k);

    auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    // printf("found best tactic \n");

    LYRA_CHECK_WITH_INFO(
        bestTactic,
        "No valid per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    m_GemmRunner->gemm(A, B, C, bias, m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), ws_size, stream);
}

template<typename DataType>
int MatmulPlugin<DataType>::initialize()
{
    configGemm();
    return 0;
}

template<typename DataType>
void MatmulPlugin<DataType>::terminate()
{
}

template<typename DataType>
size_t MatmulPlugin<DataType>::getSerializationSize()
{
    return sizeof(mDims) +                                  // Dimensions
           mPluginProfiler->getSerializationSize(mGemmId);  // selected tactics container size
}

template<typename DataType>
void MatmulPlugin<DataType>::serialize(void* buffer)
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

template<typename DataType>
void MatmulPlugin<DataType>::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

template class MatmulPlugin<half>;
// #ifdef ENABLE_BF16
// template class MatmulPlugin<__nv_bfloat16>;
// #endif

}  // namespace gemm
}  // namespace lyradiff