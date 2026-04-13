// include/graph/spfa_engine.hpp
#pragma once
#include <vector>
#include <atomic>
#include <limits>
#include <memory>
#include "csr_graph.hpp"
#include "concurrent_queue.hpp"
#include "thread_pool.hpp"

namespace graph {

struct SPFAResult {
    std::vector<std::atomic<int32_t>> distances;
    std::vector<std::atomic<uint32_t>> predecessors;
    bool success;
    
    explicit SPFAResult(size_t n) 
        : distances(n), predecessors(n), success(false) {
        for (auto& d : distances) d.store(std::numeric_limits<int32_t>::max());
        for (auto& p : predecessors) p.store(std::numeric_limits<uint32_t>::max());
    }
};

class SPFASolver {
public:
    explicit SPFASolver(const CSRGraph& graph, ThreadPool& pool);
    
    // Single-threaded scalar version for comparison/baseline
    [[nodiscard]] SPFAResult solve_scalar(uint32_t source);
    
    // AVX-512 vectorized version
    [[nodiscard]] SPFAResult solve_vectorized(uint32_t source);
    
    // Concurrent version with lock-free queue
    [[nodiscard]] SPFAResult solve_concurrent(uint32_t source, size_t num_workers = 0);

private:
    const CSRGraph& graph_;
    ThreadPool& pool_;
    
    // AVX-512 vectorized relaxation of edges
    void relax_edges_avx512(uint32_t u, 
                           std::atomic<int32_t>* distances,
                           concurrent::LockFreeQueue& queue,
                           std::atomic<bool>* in_queue);
    
    // Scalar relaxation for remaining edges (<16)
    void relax_edges_scalar(uint32_t u, uint32_t start, uint32_t end,
                           std::atomic<int32_t>* distances,
                           concurrent::LockFreeQueue& queue,
                           std::atomic<bool>* in_queue);
};

}