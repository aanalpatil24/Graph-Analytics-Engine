#pragma once
#include "baseline_graph.hpp"
#include <queue>
#include <vector>

namespace baseline {

/**
 * @brief Standard Sequential Cycle Detection (Kahn's Algorithm).
 * * BASELINE BOTTLENECK: Single-threaded execution. The queue operations 
 * and in-degree decrements are not atomic. If we tried to parallelize this 
 * with a ThreadPool, we would have to wrap the entire loop in a `std::mutex`, 
 * causing massive thread contention and context-switching overhead.
 */
inline bool has_cycle(const Graph& g) {
    std::vector<int> in_degree(g.num_vertices, 0);
    
    // Pass 1: Calculate in-degrees
    for (int i = 0; i < g.num_vertices; ++i) {
        for (const auto& edge : g.adj_list[i]) {
            in_degree[edge.dst]++;
        }
    }

    std::queue<int> q;
    for (int i = 0; i < g.num_vertices; ++i) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }

    int processed_count = 0;

    // Pass 2: Process sequentially
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        processed_count++;

        for (const auto& edge : g.adj_list[u]) {
            in_degree[edge.dst]--;
            if (in_degree[edge.dst] == 0) {
                q.push(edge.dst);
            }
        }
    }

    // If we haven't processed all vertices, a cycle exists
    return processed_count != g.num_vertices;
}

} // namespace baseline