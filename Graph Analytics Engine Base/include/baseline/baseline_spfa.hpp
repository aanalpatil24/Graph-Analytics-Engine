#pragma once
#include "baseline_graph.hpp"
#include <queue>
#include <vector>

namespace baseline {

const int INF = 1e9;

inline std::vector<int> compute_spfa(const Graph& g, int source) {
    std::vector<int> dist(g.num_vertices, INF);
    std::vector<bool> in_queue(g.num_vertices, false);
    std::queue<int> q; // Standard STL queue, highly cache-unfriendly

    dist[source] = 0;
    q.push(source);
    in_queue[source] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        in_queue[u] = false;

        // Scalar execution: processing one edge at a time.
        // In the optimized version, AVX-512 processes 16 of these simultaneously.
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