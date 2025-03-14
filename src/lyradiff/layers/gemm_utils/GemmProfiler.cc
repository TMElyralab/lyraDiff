
#include "GemmProfiler.h"
#include "src/lyradiff/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/lyradiff/kernels/cutlass_kernels/gemm/gemm.h"
#include "src/lyradiff/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"

namespace lyradiff {

namespace gemm {

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler()
{
    mMNKProfileMap = std::make_shared<MNKProfileMap>();

    // set SKIP_GEMM_PLUGIN_PROFILINGS=1 to avoid tactics profilings
    auto const skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
    mSkip              = (skipEnv != NULL && std::stoi(skipEnv));
    if (mSkip) {
        FT_LOG_DEBUG(
            "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
            "if default tactic is not defined.");
    }
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(char*&            buffer,
                                                                                  GemmIdType const& gemmId) const
{
    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    // Save number of profiles for given GEMM ID
    write(buffer, static_cast<int>(mProfileMap->size()));
    for (auto const& pair : *mProfileMap) {
        // Save pair of M to the best GEMM config
        write(buffer, pair);
    }
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(char const*&      data,
                                                                                    GemmDims&         dims,
                                                                                    GemmIdType const& gemmId)
{
    // NOTE: this mutex is not needed since each thread owns its private map, but will put here for
    // consistency
    writer_lock lock(mMNKProfileMap->mutex);

    mDims = dims;

    // GemmId gemmId(dims.n, dims.k);
    if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
        // Create GEMM with GEMM ID if it does not exist
        mMNKProfileMap->createMProfileMap(gemmId);
    }
    // Populate map with profiles of GEMM ID
    auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
    int  selectedMapSize;
    read(data, selectedMapSize);
    for (int ii = 0; ii < selectedMapSize; ++ii) {
        std::pair<int, std::optional<Config>> config;
        read(data, config);
        profileMap->insert(config);
    }
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
size_t
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(GemmIdType const& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);
    return sizeof(int) +  // size of the tactics map
           mMNKProfileMap->getMProfileMap(gemmId)->size()
               * sizeof(std::pair<int, std::optional<Config>>);  // size of the tactics map
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(RunnerPtr const&  runner,
                                                                                       GemmDims const&   dims,
                                                                                       GemmIdType const& gemmId)
{
    writer_lock lock(mMNKProfileMap->mutex);

    if (!dims.isInitialized()) {
        return;
    }

    mRunner = runner;

    int const maxM = std::min(nextPowerOfTwo(dims.maxM), MAX_PROFILE_M);
    computeTmpSize(maxM, dims.n, dims.k);

    if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
        // Create map for GEMM ID
        mMNKProfileMap->createMProfileMap(gemmId);
    } else {
        // 如果已经有当前gemm id，就不用进行配置搜索了
        return;
    }

    if (mSkip) {
        return;
    }

    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    auto profileTactics = [&mProfileMap, this](int m, int n, int k) {
        if (mProfileMap->count(m) == 0) {
            initTmpData(m, n, k, mWorkspaceTmp, mTmpWorkspaceSizeInBytes, cudaStreamDefault);
            const auto tactics = this->getTactics(m, n, k);
            // Profile different tactics for particular m and insert best config to the map
            mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
        }
    };

    // Allocate tmp data to run GEMMs
    allocateTmpData();

    int const startMinMRounded = nextPowerOfTwo(dims.minM);
    for (int m = startMinMRounded; m < maxM; m *= 2) {
        profileTactics(m, dims.n, dims.k);
    }

    profileTactics(maxM, dims.n, dims.k);
    // Free tmp data
    freeTmpData();
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(int m, GemmIdType const& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);

    if (mSkip) {
        return std::nullopt;
    }

    int const mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
    fflush(stdout);
    // printf("cur gemm id n: %d, k: %d, mRounded: %d \n", gemmId.n, gemmId.k, mRounded);
    return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData()
{
    LYRA_CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
    auto const status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
    LYRA_CHECK_WITH_INFO(status == cudaSuccess, "Can't allocate tmp workspace for GEMM tactics profiling.");
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData()
{
    auto const status = cudaFree(mWorkspaceTmp);
    LYRA_CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, std::vector<Config> const& tactics)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    float  bestTime = std::numeric_limits<float>::max();
    Config bestConfig;
    bool   foundOne = false;

    // Iterate over all tactics for given M, N and K
    for (int ii = 0; ii < tactics.size(); ++ii) {
        Config const& candidateConfig = tactics[ii];
        float         time            = std::numeric_limits<float>::max();
        try {
            if (!checkTactic(m, n, k, candidateConfig)) {
                continue;
            }
            // Profile particualar tactic for given M, N and K
            time     = profileTacticForProblem(m, n, k, candidateConfig);
            foundOne = true;
        }
        catch (std::exception const& e) {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << ii << " (for" << " m=" << m << ", n=" << n << ", k=" << k << ")"
                << ", reason: \"" << e.what() << "\". Skipped";
            FT_LOG_TRACE(msg.str());
            // printf("gemm error: Cannot profile configuration %d for m= %d, n= %d, k=%d, reason: \" %s \" Skipped \n",
            //        ii,
            //        m,
            //        n,
            //        k,
            //        e.what());
            cudaGetLastError();  // Reset the last cudaError to cudaSuccess.
            continue;
        }

        // Choose the fastest tactic
        if (time < bestTime) {
            bestConfig = candidateConfig;
            bestTime   = time;
        }
    }

    if (!foundOne) {
        std::ostringstream msg;
        msg << "Have not found any valid GEMM config for shape (" << "m=" << m << ", n=" << n << ", k=" << k
            << "). Will try to use default or fail at runtime";
        FT_LOG_WARNING(msg.str());
        printf("!foundOne error: %s \n", msg.str());

        return std::nullopt;
    }

    // printf(
    //     "found best config: with tile_config: %d, split_k_style: %d, split_k_factor: %d, stages%d, with bestTime: %f  \n",
    //     bestConfig.tile_config,
    //     bestConfig.split_k_style,
    //     bestConfig.split_k_factor,
    //     bestConfig.stages,
    //     bestTime);

    return {bestConfig};
}

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(int           m,
                                                                                                 int           n,
                                                                                                 int           k,
                                                                                                 Config const& tactic)
{
    constexpr int warmup = 5;
    constexpr int runs   = 10;

    cudaStream_t stream = cudaStreamDefault;
    // Warmup the execution
    for (int i = 0; i < warmup; ++i) {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Profile GEMM
    for (int i = 0; i < runs; ++i) {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed / runs;
}

template class GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<kernels::cutlass_kernels::CutlassGemmRunnerInterface>,
                                  GemmIdCore,
                                  GemmIdCoreHash>;

template class GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>,
                                  GemmIdCore,
                                  GemmIdCoreHash>;

template class GemmPluginProfiler<cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>,
                                  GemmIdCore,
                                  GemmIdCoreHash>;

}  // namespace gemm
}  // namespace lyradiff
