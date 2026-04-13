// src/cycle_detector.cpp
#include "graph/cycle_detector.hpp"
#include <stack>
#include <algorithm>

namespace graph {

CycleDetector::CycleDetector(const CSRGraph& graph, ThreadPool& pool) 
    : graph_(graph), pool_(pool) {}

CycleInfo CycleDetector::detect_and_resolve() {
    CycleInfo info;
    const size_t n = graph_.num_vertices();
    
    // Atomic in-degree array
    std::vector<std::atomic<uint32_t>> in_degree(n);
    for (size_t i = 0; i < n; ++i) in_degree[i].store(0);
    
    // Compute in-degrees in parallel
    ParallelExecutor executor(pool_);
    executor.parallel_for(0, n, [&](uint32_t u) {
        auto neighbors = graph_.neighbors(u);
        for (uint32_t v : neighbors) {
            in_degree[v].fetch_add(1, std::memory_order_relaxed);
        }
    });
    
    // Lock-free queue for zero in-degree nodes
    concurrent::LockFreeQueue zero_queue;
    std::vector<bool> processed(n, false);
    
    // Initial zero in-degree nodes
    for (uint32_t i = 0; i < n; ++i) {
        if (in_degree[i].load(std::memory_order_relaxed) == 0) {
            zero_queue.push(i);
        }
    }
    
    // Parallel Kahn's algorithm
    std::mutex topo_mutex;
    std::atomic<size_t> processed_count{0};
    
    auto worker = [&]() {
        while (true) {
            auto opt_u = zero_queue.pop();
            if (!opt_u) {
                // Check if done
                if (processed_count.load(std::memory_order_acquire) >= n) break;
                continue;
            }
            
            uint32_t u = *opt_u;
            if (processed[u]) continue;
            processed[u] = true;
            
            {
                std::lock_guard<std::mutex> lock(topo_mutex);
                info.topological_order.push_back(u);
            }
            processed_count.fetch_add(1);
            
            // Reduce in-degree of neighbors
            auto neighbors = graph_.neighbors(u);
            for (uint32_t v : neighbors) {
                uint32_t new_deg = in_degree[v].fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (new_deg == 0) {
                    zero_queue.push(v);
                }
            }
        }
    };
    
    // Run workers
    size_t num_workers = std::min(pool_.size(), size_t(4)); // Limit contention
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, worker));
    }
    for (auto& f : futures) f.wait();
    
    // Check for cycles
    info.has_cycle = (info.topological_order.size() != n);
    
    if (info.has_cycle) {
        // Find unvisited nodes (part of cycles)
        std::vector<bool> unvisited(n, false);
        for (uint32_t i = 0; i < n; ++i) {
            if (!processed[i]) unvisited[i] = true;
        }
        
        info.cycle_nodes = find_cycle_components(unvisited);
    }
    
    return info;
}

std::vector<uint32_t> CycleDetector::find_cycle_components(
    const std::vector<bool>& unvisited_mask) {
    
    std::vector<uint32_t> cycles;
    const size_t n = graph_.num_vertices();
    std::vector<bool> visited(n, false);
    
    // Parallel DFS from each unvisited node
    std::mutex cycle_mutex;
    
    ParallelExecutor executor(pool_);
    executor.parallel_for(0, n, [&](uint32_t start) {
        if (!unvisited_mask[start] || visited[start]) return;
        
        std::vector<uint32_t> local_stack;
        std::vector<uint32_t> local_component;
        local_stack.push_back(start);
        
        while (!local_stack.empty()) {
            uint32_t u = local_stack.back();
            local_stack.pop_back();
            
            if (visited[u]) continue;
            visited[u] = true;
            local_component.push_back(u);
            
            auto neighbors = graph_.neighbors(u);
            for (uint32_t v : neighbors) {
                if (unvisited_mask[v] && !visited[v]) {
                    local_stack.push_back(v);
                }
            }
        }
        
        if (!local_component.empty()) {
            std::lock_guard<std::mutex> lock(cycle_mutex);
            cycles.insert(cycles.end(), local_component.begin(), local_component.end());
        }
    });
    
    return cycles;
}

}