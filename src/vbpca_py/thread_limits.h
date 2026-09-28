#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace vbpca_threads {

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
