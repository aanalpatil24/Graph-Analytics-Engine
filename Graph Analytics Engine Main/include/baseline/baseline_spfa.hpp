#pragma once
#include "baseline_graph.hpp"
#include <queue>
#include <vector>

namespace baseline {

const int INF = 1e9;

/**
 * @brief Standard Scalar SPFA (Shortest Path Faster Algorithm).
 * * BASELINE BOTTLENECK: Uses a standard scalar `for` loop and a `std::queue`. 
 * The CPU must fetch, add, compare, and write exactly ONE edge per clock cycle.
 * It also suffers from branch prediction penalties during the `if (dist[u] + w < dist[v])` check.
 */
inline std::vector<int> compute_spfa(const Graph& g, int source) {
    std::vector<int> dist(g.num_vertices, INF);
    std::vector<bool> in_queue(g.num_vertices, false);
    
    // Standard STL queue is heavily cache-unfriendly for graph traversals
    std::queue<int> q; 

    dist[source] = 0;
    q.push(source);
    in_queue[source] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        in_queue[u] = false;

        // SCALAR EXECUTION: Processing one edge at a time.
        for (const auto& edge : g.adj_list[u]) {
            int v = edge.dst;
            int w = edge.weight;
            
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) {
                    q.push(v);
                    in_queue[v] = true;
                }
            }
        }
    }
    return dist;
}

} // namespace baseline