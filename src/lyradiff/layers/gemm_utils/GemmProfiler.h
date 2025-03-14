#pragma once

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "assert.h"
#include "src/lyradiff/utils/cuda_utils.h"
#include "src/lyradiff/utils/logger.h"

namespace lyradiff {

namespace gemm {

// Write values into buffer
template<typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template<typename T>
void read(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

struct GemmDims {
    int32_t minM;
    int32_t maxM;
    int32_t n;
    int32_t k;

    GemmDims(): minM(-1), maxM(-1), n(-1), k(-1) {}

    GemmDims(int32_t minM_, int32_t maxM_, int32_t n_, int32_t k_): minM(minM_), maxM(maxM_), n(n_), k(k_) {}

    bool isInitialized() const
    {
        return minM >= 0 && maxM >= 0 && n >= 0 && k >= 0;
    }
};

// Unique ID of GEMM
// In our case GEMM is uniqly identified by N and K
class GemmIdCore {
public:
    int n;
    int k;

    GemmIdCore(int n_, int k_): n(n_), k(k_) {}

    GemmIdCore(): n(-1), k(-1) {}

    bool operator==(GemmIdCore const& id) const
    {
        return isEqual(id);
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIdCore const& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        return out;
    }

protected:
    bool isEqual(GemmIdCore const& id) const
    {
        return n == id.n && k == id.k;
    }
};

// Hash of GemmId
struct GemmIdCoreHash {
    std::size_t operator()(GemmIdCore const& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        return h1 ^ h2;
    }
};

class GemmIdCublas: public GemmIdCore {
public:
    bool transA{};
    bool transB{};

    GemmIdCublas(int n_, int k_, bool transA_, bool transB_): GemmIdCore(n_, k_), transA(transA_), transB(transB_) {}

    GemmIdCublas() {}

    bool operator==(GemmIdCublas const& id) const
    {
        return isEqual(id) && transA == id.transA && transB == id.transB;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIdCublas const& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " transA=" << id.transA;
        out << " transB=" << id.transB;
        return out;
    }
};

// Hash of GemmIdCublas
struct GemmIdCublasHash {
    std::size_t operator()(GemmIdCublas const& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);

        auto h3 = std::hash<bool>{}(id.transA);
        auto h4 = std::hash<bool>{}(id.transB);

        return h1 ^ h2 ^ h3 ^ h4;
    }
};

template<typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler {
public:
    // static constexpr int MAX_PROFILE_M = 8192;
    static constexpr int MAX_PROFILE_M = 16384;

    // Map for single GEMM for different Ms (GEMM dimension) to the best config for particular M
    using MProfileMap    = std::unordered_map<int, std::optional<Config>>;
    using MProfileMapPtr = std::shared_ptr<MProfileMap>;

    // requires exclusive ownership to write to *this
    using reader_lock = std::unique_lock<std::shared_timed_mutex>;
    // requires shared ownership to read from other
    using writer_lock = std::shared_lock<std::shared_timed_mutex>;

    // Struct of continuing map if GEMMs to the best profiles for different Ms
    struct MNKProfileMap {
        // Mutex guarding map
        std::shared_timed_mutex mutex;
        // Map from GEMM Id to profile for particular GEMM
        std::unordered_map<GemmIdType, MProfileMapPtr, GemmIdHashType> profileMap;

        bool existsMProfileMap(GemmIdType const& id)
        {
            auto const iter = profileMap.find(id);
            return iter != profileMap.end();
        }

        void createMProfileMap(GemmIdType const& id)
        {
            profileMap[id] = std::make_shared<MProfileMap>();
        }

        MProfileMapPtr getMProfileMap(GemmIdType const& id)
        {
            auto const iter = profileMap.find(id);
            if (iter == profileMap.end()) {
                std::ostringstream msg;
                msg << "Cannot find ID (" << id << ") in the profile map. Abort.";
                FT_LOG_ERROR(msg.str());
            }
            return iter->second;
        }
    };

    using MNKProfileMapPtr = std::shared_ptr<MNKProfileMap>;

    GemmPluginProfiler();

    void serialize(char*& buffer, GemmIdType const& gemmId) const;

    void   deserialize(char const*& data, GemmDims& dims, GemmIdType const& gemmId);
    size_t getSerializationSize(GemmIdType const& gemmId) const;

    void profileTactics(RunnerPtr const& runner, GemmDims const& dims, GemmIdType const& gemmId);

    void setSelectionTactics(MNKProfileMapPtr const& map)
    {
        mMNKProfileMap = map;
    }

    void setTmpWorkspaceSizeInBytes(size_t bytes)
    {
        mTmpWorkspaceSizeInBytes = bytes;
    }

    void setSkip(bool skip)
    {
        mSkip = mSkip || skip;
    }

    std::optional<Config> getBestConfig(int m, GemmIdType const& gemmId) const;

protected:
    virtual void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) = 0;

    virtual void computeTmpSize(int maxM, int n, int k) = 0;

    virtual bool checkTactic(int m, int n, int k, Config const& tactic) const
    {
        return true;
    }

    virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

    virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream) {};

private:
    void allocateTmpData();

    void freeTmpData();

    std::optional<Config> profileTacticsForProblem(int m, int n, int k, std::vector<Config> const& tactics);

    float profileTacticForProblem(int m, int n, int k, Config const& tactic);

    int nextPowerOfTwo(int v) const
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return ++v;
    }

protected:
    RunnerPtr mRunner{nullptr};

private:
    MNKProfileMapPtr mMNKProfileMap{};

    size_t mTmpWorkspaceSizeInBytes{0};

    char* mWorkspaceTmp{nullptr};

    GemmDims mDims{};

    bool mSkip{false};
};

template<typename GemmPluginProfilerType>
class GemmPluginProfilerManager {
public:
    using MNKProfileMap         = typename GemmPluginProfilerType::MNKProfileMap;
    using MNKProfileMapPtr      = typename GemmPluginProfilerType::MNKProfileMapPtr;
    using GemmPluginProfilerPtr = std::shared_ptr<GemmPluginProfilerType>;

    GemmPluginProfilerManager()
    {
        mMNKProfileMap = std::make_shared<MNKProfileMap>();
    }

    GemmPluginProfilerPtr createGemmPluginProfiler(bool inference, bool skip = false)
    {
        auto profiler = std::make_shared<GemmPluginProfilerType>();
        profiler->setSkip(skip);
        // If the profiler is created during the engine build,
        // mMNKProfileMap is shared between different profilers to minimize the time spent on the profiling
        // and do not repeat profiling for the GEMMs of the same shape.
        if (!inference) {
            profiler->setSelectionTactics(mMNKProfileMap);
        }
        return profiler;
    }

private:
    MNKProfileMapPtr mMNKProfileMap{};
};

}  // namespace gemm
}  // namespace lyradiff
