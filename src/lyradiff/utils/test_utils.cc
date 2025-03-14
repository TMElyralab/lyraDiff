// PerfTime.cpp
#include "src/lyradiff/utils/test_utils.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
 
namespace lyradiff {

namespace PerfTime{
    using UINT64 = unsigned long long;
    //using namespace std::chrono;
 
    // 打点类型
    struct HitInfo {
        PerfType type;
        UINT64 timePoint;
        bool isStart = true;
 
        HitInfo(UINT64 tp, PerfType type, bool isStart) : timePoint(tp), type(type), isStart(isStart) {}
    };
 
    // 统计数据
    struct StatisticsInfo {
        UINT64 callCount;
        UINT64 costTime;
 
        StatisticsInfo(UINT64 call, UINT64 time) : callCount(call), costTime(time) {}
    };
 
    // 统计功能实现类
    class Interface {
    public:
        static void StartHit(PerfType type, const std::string &mark);
 
        static void EndHit(PerfType type, const std::string &mark);
 
        static void OutputReport(const std::string &file, bool isOutPutCmd);
 
        static UINT64 GetSysTimePoint();
 
    private:
        static std::map<std::string, StatisticsInfo> Cal();
 
        static void InsertHitInfo(PerfType type, const std::string &mark, bool isStart);
 
        static std::mutex mtRecord;
        static std::map<std::string, std::vector<HitInfo>> record;
    };
 
    // 静态数据成员定义
    std::mutex Interface::mtRecord;
    std::map<std::string, std::vector<HitInfo>> Interface::record;
 
    UINT64 Interface::GetSysTimePoint() {
        using namespace std::chrono;
        return time_point_cast<std::chrono::microseconds>(steady_clock::now()).time_since_epoch().count();
    }
 
    void Interface::InsertHitInfo(PerfType type, const std::string &mark, bool isStart) {
        auto tp = GetSysTimePoint();
        std::lock_guard<std::mutex> lck(mtRecord);
        record[mark].emplace_back(tp, type, isStart);
    }
 
    void Interface::StartHit(PerfType type, const std::string &mark) {
        InsertHitInfo(type, mark, true);
    }
 
    void Interface::EndHit(PerfType type, const std::string &mark) {
        InsertHitInfo(type, mark, false);
    }
 
    void Interface::OutputReport(const std::string &file, bool isOutPutCmd) {
        std::map<std::string, StatisticsInfo> res = Cal();
        std::ofstream ofs;
        ofs.open(file, std::ofstream::out | std::ofstream::ate);
        if (ofs.is_open()) {
            std::ostringstream oss;
            int index = 0;
            for (const auto &item: res) {
                oss << ++index << ". " << item.first << "\n" << "CallCount: " <<
                    item.second.callCount << ", Time: " <<
                    item.second.costTime << " (microseconds) | " <<
                    item.second.costTime / 1000.0 << " (milliseconds) | " <<
                    item.second.costTime / (1000.0 * 1000.0) << " (seconds)\n";
            }
            ofs << oss.str() << std::flush;
            if (isOutPutCmd) {
                std::cout << oss.str() << std::flush;
            }
        }
    }
 
    std::map<std::string, StatisticsInfo> Interface::Cal() {
        std::map<std::string, std::vector<HitInfo>> recordCp;
        {
            std::lock_guard<std::mutex> lck(mtRecord);
            recordCp = record;
        }
 
        // 循环计算每个 mark 耗时信息
        std::map<std::string, StatisticsInfo> res;
        for (const auto &oneRecord: recordCp) {
            std::vector<UINT64> startVec;
            std::vector<UINT64> endVec;
            const std::string &mark = oneRecord.first;
            const std::vector<HitInfo> &hitInfos = oneRecord.second;
            for (auto &hitInfo: hitInfos) {
                if (hitInfo.isStart) {
                    startVec.push_back(hitInfo.timePoint);
                    continue;
                }
                endVec.push_back(hitInfo.timePoint);
            }
 
            // 简单校验打点数据是否为空
            if (startVec.empty() || endVec.empty()) {
                res.insert({mark, {0, 0}});
            }
 
            // 计算耗时
            if (hitInfos.back().type == PerfType::PERF_E2E) {
                // 端到端类型
                UINT64 durTime = *(std::max_element(endVec.cbegin(), endVec.cend())) -
                                 *(std::min_element(startVec.cbegin(), startVec.cend()));
                res.insert({mark, {1, durTime}});
            } else {
                // 累加类型
                if (endVec.size() != startVec.size()) {
                    // 数据校验，开始打点和结束打点数量需要一致
                    res.insert({mark, {0, 0}});
                    continue;
                }
                UINT64 durTime = std::accumulate(endVec.cbegin(), endVec.cend(), 0) -
                                 std::accumulate(startVec.cbegin(), startVec.cend(), 0);
                res.insert({mark, {endVec.size(), durTime}});
            }
        }
        return res;
    }
 
 
    void StartHit(PerfType type, const char *mark) {
        Interface::StartHit(type, mark);
    }
 
    void EndHit(PerfType type, const char *mark) {
        Interface::EndHit(type, mark);
    }
 
    void OutputReport(const char *file, bool isOutputCmd) {
        Interface::OutputReport(file, isOutputCmd);
    }
}
}