#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace vbpca_threads {

inline int quota_at(const std::string &directory) {
    std::ifstream v2(directory + "/cpu.max");
    std::string quota;
    long long period = 0;
    if (v2 >> quota >> period) {
        if (quota == "max") return 0;
        try {
            const long long value = std::stoll(quota);
            if (value > 0 && period > 0)
                return static_cast<int>(std::max(1LL, std::min(2147483647LL, value / period)));
        } catch (...) {}
    }
    std::ifstream v1quota(directory + "/cpu.cfs_quota_us");
    std::ifstream v1period(directory + "/cpu.cfs_period_us");
    long long value = 0;
    if (v1quota >> value && v1period >> period && value > 0 && period > 0)
        return static_cast<int>(std::max(1LL, std::min(2147483647LL, value / period)));
    return 0;
}

inline int cgroup_limit() {
#if defined(__linux__)
    int limit = 0;
    auto include = [&](const std::string &directory) {
        const int quota = quota_at(directory);
        if (quota > 0) limit = limit > 0 ? std::min(limit, quota) : quota;
    };
    for (const char *mount : {"/sys/fs/cgroup", "/sys/fs/cgroup/cpu",
            "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct,cpu"}) include(mount);
    std::ifstream membership("/proc/self/cgroup");
    std::string line;
    while (std::getline(membership, line)) {
        const auto first = line.find(':');
        const auto second = line.find(':', first == std::string::npos ? 0 : first + 1);
        if (first == std::string::npos || second == std::string::npos) continue;
        const std::string controllers = "," + line.substr(first + 1, second - first - 1) + ",";
        const std::string group = line.substr(second + 1);
        if (group.empty() || group[0] != '/' || group.find("..") != std::string::npos) continue;
        for (const char *mount : {"/sys/fs/cgroup", "/sys/fs/cgroup/cpu",
                "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct,cpu"}) {
            if ((controllers == ",," && std::string(mount) != "/sys/fs/cgroup") ||
                (controllers != ",," && (controllers.find(",cpu,") == std::string::npos ||
                 std::string(mount) == "/sys/fs/cgroup"))) continue;
            std::string relative = group;
            while (!relative.empty()) {
                include(std::string(mount) + relative);
                relative.erase(relative.find_last_of('/'));
            }
        }
    }
    return limit;
#else
    return 0;
#endif
}

inline int positive_env(const char *key) {
    const char *raw = std::getenv(key);
    if (raw == nullptr) {
        return 0;
    }
    try {
        const int parsed = std::stoi(raw);
        return parsed > 0 ? parsed : 0;
    } catch (...) {
        return 0;
    }
}

inline int process_thread_limit() {
    const unsigned int hardware = std::thread::hardware_concurrency();
    int limit = hardware > 0 ? static_cast<int>(hardware) : 1;

#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        int affinity_count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &affinity)) {
                ++affinity_count;
            }
        }
        if (affinity_count > 0) {
            limit = std::min(limit, affinity_count);
        }
    }
#endif

    for (const char *key : {"SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"}) {
        const int allocated = positive_env(key);
        if (allocated > 0) {
            limit = std::min(limit, allocated);
        }
    }
    const int quota = cgroup_limit();
    if (quota > 0) limit = std::min(limit, quota);
    return std::max(1, limit);
}

inline int resolve_thread_count(
    int requested,
    int n_items,
    const char *specific_env = nullptr
) {
    const int allocation = process_thread_limit();
    int desired = requested;
    if (desired <= 0 && specific_env != nullptr) {
        desired = positive_env(specific_env);
    }
    if (desired <= 0) {
        desired = positive_env("VBPCA_NUM_THREADS");
    }
    if (desired <= 0) {
        desired = allocation;
    }
    return std::max(1, std::min({desired, allocation, std::max(1, n_items)}));
}

inline bool has_thread_override(int requested, const char *specific_env = nullptr) {
    if (requested > 0) {
        return true;
    }
    if (specific_env != nullptr && positive_env(specific_env) > 0) {
        return true;
    }
    return positive_env("VBPCA_NUM_THREADS") > 0;
}

}  // namespace vbpca_threads
