
#include "lyraException.h"

#include <cstdlib>
#if !defined(_MSC_VER)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif
#include <sstream>

namespace lyradiff {

namespace {
int constexpr VOID_PTR_SZ = 2 + sizeof(void*) * 2;
}

#if !defined(_MSC_VER)

LyraException::LyraException(char const* file, std::size_t line, const std::string& msg): std::runtime_error{""}
{
    mNbFrames        = backtrace(mCallstack.data(), MAX_FRAMES);
    auto const trace = getTrace();
    std::runtime_error::operator=(
        std::runtime_error{fmtstr("%s (%s:%zu)\n%s", msg.c_str(), file, line, trace.c_str())});
}
#else
LyraException::LyraException(char const* file, std::size_t line, const std::string& msg):
    mNbFrames{}, std::runtime_error{fmtstr("%s (%s:%zu)", msg.c_str(), file, line)}
{
}
#endif

LyraException::~LyraException() = default;

std::string LyraException::getTrace() const
{
#if defined(_MSC_VER)
    return "";
#else
    auto const         trace = backtrace_symbols(mCallstack.data(), mNbFrames);
    std::ostringstream buf;
    for (auto i = 1; i < mNbFrames; ++i) {
        Dl_info info;
        if (dladdr(mCallstack[i], &info) && info.dli_sname) {
            auto const clearName = demangle(info.dli_sname);
            buf << fmtstr("%-3d %*p %s + %zd",
                          i,
                          VOID_PTR_SZ,
                          mCallstack[i],
                          clearName.c_str(),
                          static_cast<char*>(mCallstack[i]) - static_cast<char*>(info.dli_saddr));
        }
        else {
            buf << fmtstr("%-3d %*p %s", i, VOID_PTR_SZ, mCallstack[i], trace[i]);
        }
        if (i < mNbFrames - 1)
            buf << std::endl;
    }

    if (mNbFrames == MAX_FRAMES)
        buf << std::endl << "[truncated]";

    std::free(trace);
    return buf.str();
#endif
}

std::string LyraException::demangle(char const* name)
{
#if defined(_MSC_VER)
    return name;
#else
    std::string clearName{name};
    auto        status    = -1;
    auto const  demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        clearName = demangled;
        std::free(demangled);
    }
    return clearName;
#endif
}

}  // namespace lyradiff
