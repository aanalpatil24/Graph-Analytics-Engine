// include/graph/thread_pool.hpp
#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "concurrent_queue.hpp"

namespace graph {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    [[nodiscard]] size_t size() const noexcept { return workers_.size(); }
    
    // Stop processing
    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
};

// Parallel executor for graph algorithms
class ParallelExecutor {
public:
    explicit ParallelExecutor(ThreadPool& pool) : pool_(pool) {}

    template<typename IndexFunc>
    void parallel_for(uint32_t start, uint32_t end, IndexFunc&& func);

    template<typename RangeFunc>
    void parallel_range(uint32_t start, uint32_t end, RangeFunc&& func);

private:
    ThreadPool& pool_;
};

}