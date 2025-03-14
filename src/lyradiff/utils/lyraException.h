
#pragma once

#include "string_utils.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#define NEW_LYRA_EXCEPTION(...) lyradiff::LyraException(__FILE__, __LINE__, lyradiff::fmtstr(__VA_ARGS__))

namespace lyradiff {

class LyraException: public std::runtime_error {
public:
    static auto constexpr MAX_FRAMES = 128;

    explicit LyraException(char const* file, std::size_t line, std::string const& msg);

    ~LyraException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const* name);

private:
    std::array<void*, MAX_FRAMES> mCallstack{};
    int                           mNbFrames;
};

}  // namespace lyradiff
