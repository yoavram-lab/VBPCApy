#pragma once

#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vbpca_threads {

// Declared after its pool so unwinding joins workers before destroying threads.
class ThreadJoiner {
public:
    explicit ThreadJoiner(std::vector<std::thread> &threads) : threads_(threads) {}
    ThreadJoiner(const ThreadJoiner &) = delete;
    ThreadJoiner &operator=(const ThreadJoiner &) = delete;
    ~ThreadJoiner() {
        for (auto &thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
private:
    std::vector<std::thread> &threads_;
};

// Workers use native data only. The caller owns any GIL-release scope.
template <typename Worker>
void parallel_ranges(int size, int count, Worker worker) {
    if (count <= 1) {
        worker(0, size);
        return;
    }
    std::exception_ptr error;
    std::mutex error_mutex;
    auto guarded = [&](int begin, int end) {
        try {
            worker(begin, end);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = std::current_exception();
            }
        }
    };
    std::vector<std::thread> pool;
    ThreadJoiner joining(pool);
    pool.reserve(static_cast<std::size_t>(count));
    int begin = 0;
    for (int i = 0; i < count; ++i) {
        const int end = begin + size / count + (i < size % count ? 1 : 0);
        pool.emplace_back(guarded, begin, end);
        begin = end;
    }
    for (auto &thread : pool) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

} // namespace vbpca_threads
