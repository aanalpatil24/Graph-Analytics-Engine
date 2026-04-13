// include/algorithms/concurrent_toposort.hpp
#pragma once
#include <vector>
#include <atomic>
#include <optional>
#include <algorithm>
#include "graph/csr_graph.hpp"
#include "utils/thread_pool.hpp"
#include "utils/lock_free_queue.hpp"

namespace graph {

/**
 * Concurrent Topological Sort using Kahn's Algorithm
 * Detects cycles and isolates cycle components using parallel DFS
 */
class ConcurrentTopologicalSort {
private:
    const CSRGraph& graph_;
    std::vector<std::atomic<int32_t>> in_degree_;
    std::vector<std::atomic<bool>> visited_;
    std::vector<VertexId> topological_order_;
    std::mutex result_mutex_;
    LockFreeQueue<VertexId> zero_in_degree_;
    std::atomic<size_t> processed_count_{0};

public:
    explicit ConcurrentTopologicalSort(const CSRGraph& graph)
        : graph_(graph)
        , in_degree_(graph.num_vertices())
        , visited_(graph.num_vertices()) {
        
        // Initialize in-degrees atomically
        for (VertexId v = 0; v < static_cast<VertexId>(graph.num_vertices()); ++v) {
            in_degree_[v].store(0, std::memory_order_relaxed);
            visited_[v].store(false, std::memory_order_relaxed);
        }
        
        // Count in-degrees (parallelizable for large graphs)
        for (VertexId v = 0; v < static_cast<VertexId>(graph.num_vertices()); ++v) {
            for (VertexId dst : graph.neighbors(v)) {
                in_degree_[dst].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    struct Result {
        std::vector<VertexId> order;
        std::vector<std::vector<VertexId>> cycles; // Isolated cycle components
        bool has_cycle;
    };

    Result compute(ThreadPool& pool) {
        // Enqueue initial zero in-degree vertices
        for (VertexId v = 0; v < static_cast<VertexId>(graph_.num_vertices()); ++v) {
            if (in_degree_[v].load(std::memory_order_acquire) == 0) {
                zero_in_degree_.enqueue(v);
                visited_[v].store(true, std::memory_order_release);
            }
        }
        
        // Process using thread pool
        std::vector<std::future<void>> futures;
        size_t num_threads = pool.size();
        
        for (size_t t = 0; t < num_threads; ++t) {
            futures.push_back(pool.submit([this]() { worker_loop(); }));
        }
        
        for (auto& f : futures) f.wait();
        
        Result result;
        result.order = std::move(topological_order_);
        
        // Check for cycles
        if (processed_count_.load(std::memory_order_acquire) < graph_.num_vertices()) {
            result.has_cycle = true;
            result.cycles = detect_cycles_parallel(pool);
        } else {
            result.has_cycle = false;
        }
        
        return result;
    }

private:
    void worker_loop() {
        while (true) {
            auto opt_v = zero_in_degree_.dequeue();
            if (!opt_v) {
                if (processed_count_.load(std::memory_order_acquire) >= graph_.num_vertices()) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            
            VertexId u = *opt_v;
            
            {
                std::lock_guard lock(result_mutex_);
                topological_order_.push_back(u);
            }
            processed_count_.fetch_add(1, std::memory_order_release);
            
            // Process neighbors
            auto neighbors = graph_.neighbors(u);
            for (VertexId v : neighbors) {
                int32_t new_deg = in_degree_[v].fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (new_deg == 0) {
                    visited_[v].store(true, std::memory_order_release);
                    zero_in_degree_.enqueue(v);
                }
            }
        }
    }

    std::vector<std::vector<VertexId>> detect_cycles_parallel(ThreadPool& pool) {
        // Find unvisited nodes (part of cycles)
        std::vector<VertexId> cycle_nodes;
        for (VertexId v = 0; v < static_cast<VertexId>(graph_.num_vertices()); ++v) {
            if (!visited_[v].load(std::memory_order_acquire)) {
                cycle_nodes.push_back(v);
            }
        }
        
        // Parallel DFS to find strongly connected components (cycles)
        std::vector<std::vector<VertexId>> cycles;
        std::mutex cycle_mutex;
        std::atomic<size_t> idx{0};
        
        std::vector<std::future<void>> futures;
        for (size_t t = 0; t < pool.size(); ++t) {
            futures.push_back(pool.submit([&]() {
                std::vector<VertexId> local_stack;
                std::vector<bool> local_visited(graph_.num_vertices(), false);
                
                while (true) {
                    size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                    if (i >= cycle_nodes.size()) break;
                    
                    VertexId start = cycle_nodes[i];
                    if (local_visited[start]) continue;
                    
                    std::vector<VertexId> component;
                    local_stack.push_back(start);
                    
                    while (!local_stack.empty()) {
                        VertexId u = local_stack.back();
                        local_stack.pop_back();
                        
                        if (local_visited[u]) continue;
                        local_visited[u] = true;
                        component.push_back(u);
                        
                        for (VertexId v : graph_.neighbors(u)) {
                            if (!visited_[v].load(std::memory_order_acquire) && !local_visited[v]) {
                                local_stack.push_back(v);
                            }
                        }
                    }
                    
                    if (!component.empty()) {
                        std::lock_guard lock(cycle_mutex);
                        cycles.push_back(std::move(component));
                    }
                }
            });
        }
        
        for (auto& f : futures) f.wait();
        return cycles;
    }
};

} // namespace graph