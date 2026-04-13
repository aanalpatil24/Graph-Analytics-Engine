// src/thread_pool.cpp
#include "graph/thread_pool.hpp"

namespace graph {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) return;
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_.store(true);
    }
    condition_.notify_all();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}

// Explicit instantiations
template auto ThreadPool::enqueue<std::function<void()>>(std::function<void()>&&) -> std::future<void>;

template<typename IndexFunc>
void ParallelExecutor::parallel_for(uint32_t start, uint32_t end, IndexFunc&& func) {
    size_t n = end - start;
    size_t num_threads = pool_.size();
    size_t chunk = (n + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        uint32_t s = start + t * chunk;
        uint32_t e = std::min(start + (t + 1) * chunk, end);
        if (s < e) {
            futures.push_back(pool_.enqueue([func, s, e]() {
                for (uint32_t i = s; i < e; ++i) {
                    func(i);
                }
            }));
        }
    }
    
    for (auto& f : futures) f.wait();
}

template<typename RangeFunc>
void ParallelExecutor::parallel_range(uint32_t start, uint32_t end, RangeFunc&& func) {
    size_t n = end - start;
    size_t num_threads = pool_.size();
    size_t chunk = (n + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        uint32_t s = start + t * chunk;
        uint32_t e = std::min(start + (t + 1) * chunk, end);
        if (s < e) {
            futures.push_back(pool_.enqueue([func, s, e]() {
                func(s, e);
            }));
        }
    }
    
    for (auto& f : futures) f.wait();
}

// Explicit instantiations
template void ParallelExecutor::parallel_for<std::function<void(uint32_t)>>(uint32_t, uint32_t, std::function<void(uint32_t)>&&);
template void ParallelExecutor::parallel_range<std::function<void(uint32_t, uint32_t)>>(uint32_t, uint32_t, std::function<void(uint32_t, uint32_t)>&&);

}