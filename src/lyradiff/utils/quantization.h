#pragma once

#include "Tensor.h"
#include "stdlib.h"
#include <optional>

namespace lyradiff {

class QuantMode {
    // [WARNING] KEEP BELOW DEFINITION IN SYNC WITH lyradiff/quantization/mode.py
public:
    using BaseType = std::uint32_t;

    explicit constexpr QuantMode(BaseType value) noexcept: mValue{value} {}

    QuantMode() noexcept = default;

    constexpr QuantMode(QuantMode const&) noexcept = default;

    constexpr QuantMode& operator=(QuantMode const& other) noexcept = default;

    static constexpr QuantMode none() noexcept
    {
        return QuantMode(BaseType(0));
    }

    static constexpr QuantMode int4Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 0);
    }

    static constexpr QuantMode int8Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 1);
    }

    static constexpr QuantMode activations() noexcept
    {
        return QuantMode(BaseType(1u) << 2);
    }

    static constexpr QuantMode perChannelScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 3);
    }

    static constexpr QuantMode perTokenScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 4);
    }

    static constexpr QuantMode perGroupScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 5);
    }

    static constexpr QuantMode int8KvCache() noexcept
    {
        return QuantMode(BaseType(1u) << 6);
    }

    static constexpr QuantMode fp8KvCache() noexcept
    {
        return QuantMode(BaseType(1u) << 7);
    }

    static constexpr QuantMode fp8Qdq() noexcept
    {
        return QuantMode(BaseType(1u) << 8);
    }

    constexpr BaseType value() const noexcept
    {
        return mValue;
    }

    constexpr bool isSet(QuantMode const& mode) const noexcept
    {
        return (mValue & mode.value()) == mode.value();
    }

    constexpr bool hasInt4Weights() const noexcept
    {
        return isSet(int4Weights());
    }

    constexpr bool hasInt8Weights() const noexcept
    {
        return isSet(int8Weights());
    }

    constexpr bool hasActivations() const noexcept
    {
        return isSet(activations());
    }

    constexpr bool hasPerChannelScaling() const noexcept
    {
        return isSet(perChannelScaling());
    }

    constexpr bool hasPerTokenScaling() const noexcept
    {
        return isSet(perTokenScaling());
    }

    constexpr bool hasPerGroupScaling() const noexcept
    {
        return isSet(perGroupScaling());
    }

    constexpr bool hasStaticActivationScaling() const noexcept
    {
        return !hasPerTokenScaling();
    }

    constexpr bool hasInt8KvCache() const noexcept
    {
        return isSet(int8KvCache());
    }

    constexpr bool hasFp8KvCache() const noexcept
    {
        return isSet(fp8KvCache());
    }

    constexpr bool hasFp8Qdq() const noexcept
    {
        return isSet(fp8Qdq());
    }

    constexpr bool hasKvCacheQuant() const noexcept
    {
        return hasInt8KvCache() || hasFp8KvCache();
    }

    static constexpr QuantMode fromDescription(bool quantizeWeights     = false,
                                               bool quantizeActivations = false,
                                               bool perToken            = false,
                                               bool perChannel          = false,
                                               bool perGroup            = false,
                                               bool useInt4Weights      = false,
                                               bool useInt8KvCache      = false,
                                               bool useFp8KvCache       = false,
                                               bool useFp8Qdq           = false)
    {
        QuantMode quantMode{};
        if (quantizeWeights) {
            if (useInt4Weights)
                quantMode += int4Weights();
            else
                quantMode += int8Weights();
        }

        if (quantizeActivations) {
            quantMode += activations();
        }

        if (perChannel) {
            quantMode += QuantMode::perChannelScaling();
        }
        if (perToken) {
            quantMode += QuantMode::perTokenScaling();
        }
        if (perGroup) {
            quantMode += QuantMode::perGroupScaling();
        }

        if (useInt8KvCache) {
            quantMode += int8KvCache();
        }

        if (useFp8KvCache) {
            quantMode += fp8KvCache();
        }

        if (useFp8Qdq) {
            quantMode += fp8Qdq();
        }

        return quantMode;
    }

    static constexpr QuantMode useSmoothQuant(bool perToken = false, bool perChannel = false)
    {
        return fromDescription(true, true, perToken, perChannel);
    }

    static constexpr QuantMode useWeightOnly(bool useInt4Weights = false, bool perGroup = false)
    {
        return fromDescription(true, false, false, false, perGroup, useInt4Weights);
    }

    static const QuantMode fromQuantAlgo(std::optional<std::string> quantAlgo        = std::nullopt,
                                         std::optional<std::string> kvCacheQuantAlgo = std::nullopt)
    {
        QuantMode quantMode{};
        if (quantAlgo == "W8A16") {
            quantMode = useWeightOnly(false, false);
        }
        else if (quantAlgo == "W4A16") {
            quantMode = useWeightOnly(true, false);
        }
        else if (quantAlgo == "W4A16_AWQ") {
            quantMode = useWeightOnly(true, true);
        }
        else if (quantAlgo == "W4A8_AWQ") {
            quantMode = useWeightOnly(true, true);
        }
        else if (quantAlgo == "W4A16_GPTQ") {
            quantMode = useWeightOnly(true, true);
        }
        else if (quantAlgo == "W8A8_SQ_PER_CHANNEL") {
            quantMode = useSmoothQuant(false, true);
        }
        else if (quantAlgo == "W8A8_SQ_PER_TENSOR_PLUGIN") {
            quantMode = useSmoothQuant(false, false);
        }
        else if (quantAlgo == "W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN") {
            quantMode = useSmoothQuant(true, true);
        }
        else if (quantAlgo == "W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN") {
            quantMode = useSmoothQuant(false, true);
        }
        else if (quantAlgo == "W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN") {
            quantMode = useSmoothQuant(true, false);
        }
        else if (quantAlgo == "FP8") {
            quantMode = fromDescription(false, false, false, false, false, false, false, false, true);
        }

        if (kvCacheQuantAlgo == "INT8") {
            quantMode += int8KvCache();
        }
        else if (kvCacheQuantAlgo == "FP8") {
            quantMode += fp8KvCache();
        }

        return quantMode;
    }

    constexpr QuantMode operator+(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue | other.mValue);
    }

    constexpr QuantMode& operator+=(QuantMode const& other) noexcept
    {
        return *this = *this + other;
    }

    constexpr QuantMode operator-(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue & ~other.mValue);
    }

    constexpr QuantMode& operator-=(QuantMode const& other) noexcept
    {
        return *this = *this - other;
    }

    constexpr bool operator==(QuantMode const& other) const noexcept
    {
        return mValue == other.mValue;
    }

    constexpr bool operator!=(QuantMode const& other) const noexcept
    {
        return !(*this == other);
    }

private:
    BaseType mValue{0};
};

}  // namespace lyradiff
