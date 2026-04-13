// include/graph/cycle_detector.hpp
#pragma once
#include <vector>
#include <atomic>
#include <optional>
#include "csr_graph.hpp"
#include "thread_pool.hpp"

namespace graph {

struct CycleInfo {
    bool has_cycle;
    std::vector<uint32_t> cycle_nodes;
    std::vector<uint32_t> topological_order;
};

class CycleDetector {
public:
    explicit CycleDetector(const CSRGraph& graph, ThreadPool& pool);
    
    // Concurrent Kahn's algorithm with atomic in-degrees
    [[nodiscard]] CycleInfo detect_and_resolve();
    
    // Parallel DFS to isolate cycles in remaining subgraph
    [[nodiscard]] std::vector<uint32_t> find_cycle_components(
        const std::vector<bool>& unvisited_mask);

private:
    const CSRGraph& graph_;
    ThreadPool& pool_;
    
    void parallel_dfs_collect(uint32_t start, 
                             std::vector<bool>& visited,
                             std::vector<uint32_t>& component);
};

}