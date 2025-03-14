
#include "SmoothQuantGemm.h"
#include "src/lyradiff/kernels/basic_transformer/int8SQ.h"

namespace lyradiff {

namespace gemm {

void SmoothQuantGemmPluginProfiler::runTactic(int                                          m,
                                              int                                          n,
                                              int                                          k,
                                              SmoothQuantGemmPluginProfiler::Config const& tactic,
                                              char*                                        workspace,
                                              cudaStream_t const&                          stream)
{
    int8_t* aTmp        = reinterpret_cast<int8_t*>(workspace);
    int8_t* bTmp        = nextWorkspacePtr(aTmp, m * k * sizeof(int8_t));
    void*   cTmp        = reinterpret_cast<void*>(nextWorkspacePtr(bTmp, n * k * sizeof(int8_t)));
    float*  alphaRowTmp = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(cTmp), m * n * 2));
    float*  alphaColTmp =
        reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaRowTmp), m * sizeof(float)));
    char* workspaceTmp =
        reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaColTmp), n * sizeof(float)));

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(
        aTmp, bTmp, mQuantMode, alphaColTmp, alphaRowTmp, cTmp, m, n, k, tactic, workspaceTmp, wsSize, stream);
}

void SmoothQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(int8_t),             // A
        n * k * sizeof(int8_t),                // B
        maxM * n * 2u,                         // C
        maxM * sizeof(float),                  // alphaRow
        n * sizeof(float),                     // alphaCol
        mRunner->getWorkspaceSize(maxM, n, k)  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<SmoothQuantGemmPluginProfiler::Config> SmoothQuantGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

template<typename DataType>
SmoothQuantGemmPlugin<DataType>::SmoothQuantGemmPlugin(
    const QuantMode& quantMode, SmoothQuantGemmPlugin<DataType>::PluginProfilerPtr const& pluginProfiler):
    mPluginProfiler(pluginProfiler)
{
    mQuantMode = quantMode;
    init();
}

template<typename DataType>
SmoothQuantGemmPlugin<DataType>::SmoothQuantGemmPlugin(const SmoothQuantGemmPlugin& plugin):
    mPluginProfiler(plugin.mPluginProfiler)
{
    mQuantMode = plugin.mQuantMode;
    init();
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::init()
{
    mArch = getSMVersion();

    m_sqGemmRunner = std::make_shared<kernels::cutlass_kernels::CutlassInt8GemmRunner<DataType>>();

    mGemmId = GemmIdCore(mDims.n, mDims.k);
}

// IPluginV2DynamicExt Methods
template<typename DataType>
SmoothQuantGemmPlugin<DataType>* SmoothQuantGemmPlugin<DataType>::clone()
{
    auto* plugin = new SmoothQuantGemmPlugin(*this);
    return plugin;
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::configGemm()
{
    mPluginProfiler->profileTactics(m_sqGemmRunner, mDims, mGemmId);
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::configurePlugin(int maxM, int K, int N)
{

    mDims   = {1, maxM, N, K};
    mGemmId = {N, K};

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(maxM, N, K);
}

template<typename DataType>
size_t SmoothQuantGemmPlugin<DataType>::getWorkspaceSize(int M, int K, int N)
{
    return m_sqGemmRunner->getWorkspaceSize(M, N, K);
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::RunGemm(int8_t const* A,
                                              int8_t const* B,
                                              float const*  alphaCol,
                                              float const*  alphaRow,
                                              void*         C,
                                              int           m,
                                              int           n,
                                              int           k,
                                              char*         workspacePtr,
                                              cudaStream_t  stream)
{
#ifdef ENABLE_BF16
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value || std::is_same<DataType, __nv_bfloat16>::value),
                         "No valid weightOnlyQuantMatmul configuration");
#else
    LYRA_CHECK_WITH_INFO((std::is_same<DataType, half>::value), "No valid weightOnlyQuantMatmul configuration");
#endif
    // if (m <= 4) {
    //     lyradiff::smooth_quant::Params params(reinterpret_cast<int8_t const*>(A),
    //                                         reinterpret_cast<int8_t const*>(B),
    //                                         reinterpret_cast<float const*>(alphaRow),
    //                                         reinterpret_cast<float const*>(alphaCol),
    //                                         reinterpret_cast<void*>(C),
    //                                         m,
    //                                         n,
    //                                         k,
    //                                         mQuantMode);
    //     lyradiff::smooth_quant::int8_sq_launcher<DataType>(params, stream);
    // }
    // else {
    int const ws_size = m_sqGemmRunner->getWorkspaceSize(m, n, k);

    GemmIdCore cur_gemm_id = {n, k};

    auto const& bestTactic = mPluginProfiler->getBestConfig(m, cur_gemm_id);
    LYRA_CHECK_WITH_INFO(
        bestTactic,
        "No valid SmoothQuantGemm per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
        "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
        "engine.)");

    m_sqGemmRunner->gemm(A,
                         B,
                         mQuantMode,
                         alphaCol,
                         alphaRow,
                         C,
                         m,
                         n,
                         k,
                         *bestTactic,
                         reinterpret_cast<char*>(workspacePtr),
                         ws_size,
                         stream);
    // }
}

template<typename DataType>
int SmoothQuantGemmPlugin<DataType>::initialize()
{
    configGemm();
    return 0;
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::terminate()
{
}

template<typename DataType>
size_t SmoothQuantGemmPlugin<DataType>::getSerializationSize()
{
    return sizeof(unsigned int) +                           // QuantMode
           sizeof(mDims) +                                  // Dimensions
           mPluginProfiler->getSerializationSize(mGemmId);  // selected tactics container size
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::serialize(void* buffer)
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

template<typename DataType>
void SmoothQuantGemmPlugin<DataType>::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

template class SmoothQuantGemmPlugin<half>;
#ifdef ENABLE_BF16
template class SmoothQuantGemmPlugin<__nv_bfloat16>;
#endif

}  // namespace gemm
}  // namespace lyradiff