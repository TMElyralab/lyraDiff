#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>   // std::make_unique
#include <sstream>  // std::stringstream
#include <string>
#include <unistd.h>
#include <vector>

#include <cstdlib>

namespace lyradiff {

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{
    // This function came from a code snippet in stackoverflow under cc-by-1.0
    //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    // Disable format-security warning in this function.
#if defined(_MSC_VER)  // for visual studio
#pragma warning(push)
#pragma warning(warning(disable : 4996))
#elif defined(__GNUC__) || defined(__clang__)  // for gcc or clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf  = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

template<typename T>
inline std::string vec2str(std::vector<T> vec)
{
    std::stringstream ss;
    ss << "(";
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            ss << vec[i] << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

template<typename T>
inline std::string arr2str(T* arr, size_t size)
{
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i] << ", ";
    }
    if (size > 0) {
        ss << arr[size - 1];
    }
    ss << ")";
    return ss.str();
}

inline std::string filepath2WeightId(const std::string& ins)
{
    // get basename
    std::string basename = ins.substr(ins.find_last_of("/") + 1);
    return basename.substr(0, ins.size() - 4);
}

inline std::string weightId2Filepath(const std::string& ss, const std::string& dirname)
{
    // get shape from a line like: downblocks.2.resnets.0.conv1.weight:shape0.shape1.shape2.shape3
    if (dirname[dirname.size() - 1] != '/') {
        return dirname + "/" + ss + ".bin";
    }
    else {
        return dirname + ss + ".bin";
    }
}

inline void weightRec2Shape(const std::string& ss, std::vector<size_t>& shape)
{
    // get shape from a line like: downblocks.2.resnets.0.conv1.weight:shape0.shape1.shape2.shape3
    std::string s;
    s.assign(ss);
    shape.clear();

    size_t start = s.find_first_of(':');
    s            = s.substr(start + 1);
    size_t pos   = 0;
    while ((pos = s.find('.')) != std::string::npos) {
        size_t x = stoi(s.substr(0, pos));
        shape.emplace_back(x);
        s.erase(0, pos + 1);
    }
    shape.emplace_back(stoi(s));
}

inline std::string weightRec2WeightId(const std::string& ss)
{
    // get shape from a line like: downblocks.2.resnets.0.conv1.weight:shape0.shape1.shape2.shape3
    size_t end = ss.find_first_of(":");
    return ss.substr(0, end);
}

inline std::string genRandomStr(const int len)
{
    static const char alphanum[] = "0123456789"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz";
    std::string       tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
}

inline bool getBoolEnvVar(const char* envVarName, bool defaultValue)
{
    // 获取环境变量的字符串表示
    const char* envVarValue = std::getenv(envVarName);

    // 默认情况下，如果没有找到环境变量，返回默认值
    if (!envVarValue) {
        return defaultValue;
    }

    // 将字符串转换为小写以忽略大小写差异
    std::string value(envVarValue);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });

    // 检查字符串是否表示真值
    return value == "true" || value == "1" || value == "yes" || value == "on";
}

}  // namespace lyradiff
