#include/utils/thread_pool.hpp
#pragma once
#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stop_token>
#include <type_traits>
#include "graphs/types.hpp"

namespace graph {

/**
 * C++20 Thread Pool with work stealing and jthread support
 * Optimized for graph algorithm parallelism
 */
class ThreadPool {
private:
    struct WorkItem {
        std::function<void()> func;
        
        template<typename F>
        WorkItem(F&& f) : func(std::forward<F>(f)) {}
    };

    std::vector<std::jthread> workers_;
    std::queue<WorkItem> global_queue_;
    std::mutex queue_mutex_;
    std::condition_variable_any cv_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};

public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this](std::stop_token st) {
                worker_loop(st);
            });
        }
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_release);
        cv_.notify_all();
        // jthreads automatically join on destruction
    }

    // Submit work to the pool
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind_front(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<ReturnType> result = task->get_future();
        
        {
            std::unique_lock lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot submit to stopped thread pool");
            }
            global_queue_.emplace([task]() { (*task)(); });
            active_tasks_.fetch_add(1, std::memory_order_relaxed);
        }
        
        cv_.notify_one();
        return result;
    }

    void wait_all() {
        while (active_tasks_.load(std::memory_order_acquire) > 0 || !global_queue_.empty()) {
            std::this_thread::yield();
        }
    }

    [[nodiscard]] size_t size() const noexcept { return workers_.size(); }

private:
    void worker_loop(std::stop_token st) {
        while (!st.stop_requested()) {
            std::optional<WorkItem> item;
            
            {
                std::unique_lock lock(queue_mutex_);
                cv_.wait(lock, st, [this] {
                    return !global_queue_.empty() || stop_.load(std::memory_order_acquire);
                });
                
                if (stop_.load(std::memory_order_acquire)) break;
                
                if (!global_queue_.empty()) {
                    item = std::move(global_queue_.front());
                    global_queue_.pop();
                }
            }
            
            if (item) {
                item->func();
                active_tasks_.fetch_sub(1, std::memory_order_release);
            }
        }
    }
};

} // namespace graph