#pragma once

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "3rdparty/cpp-httplib/httplib.h"
#include "3rdparty/json/single_include/nlohmann/json.hpp"
#include <ctime>
#include <iostream>

using namespace std;

using json = nlohmann::json;

namespace lyradiff {

inline std::string registerMachine(const std::string& secretId,
                                   const std::string& secretKey,
                                   const std::string& machineId,
                                   const bool         is_internal = false)
{
    return "";
}

inline int deRegisterMachine(const std::string& token, const bool is_internal = false)
{
    return -1;
}

inline std::string heartbeatMachine(const std::string& token, const bool is_internal = false)
{
    return "";
}

}  // namespace lyradiff