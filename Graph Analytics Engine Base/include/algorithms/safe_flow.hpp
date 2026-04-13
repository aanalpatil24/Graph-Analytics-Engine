// include/algorithms/safe_flow.hpp
#pragma once
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include "graph/csr_graph.hpp"
#include "algorithms/spfa_avx512.hpp"

namespace graph {

/**
 * Safe Flow Decomposition - Identifies maximal safe paths in Directed Flow Networks
 * Uses modified SPFA to find paths that don't violate capacity constraints
 */
class SafeFlowDecomposition {
private:
    const CSRGraph& graph_;
    std::vector<Weight> capacities_;
    
public:
    struct Path {
        std::vector<VertexId> vertices;
        Weight flow;
        Weight min_capacity;
    };

    explicit SafeFlowDecomposition(const CSRGraph& graph, std::vector<Weight> capacities)
        : graph_(graph), capacities_(std::move(capacities)) {}

    /**
     * Find all maximal safe paths from source to sink
     * A path is safe if flow <= min edge capacity along the path
     */
    std::vector<Path> find_safe_paths(VertexId source, VertexId sink) {
        std::vector<Path> safe_paths;
        std::vector<bool> blocked(graph_.num_vertices(), false);
        
        // Iteratively find augmenting paths using SPFA with capacity constraints
        while (true) {
            auto path = find_augmenting_path(source, sink, blocked);
            if (path.vertices.empty()) break;
            
            // Validate path safety
            if (is_path_safe(path)) {
                safe_paths.push_back(path);
                // Block vertices to find edge-disjoint paths
                for (VertexId v : path.vertices) {
                    if (v != source && v != sink) {
                        blocked[v] = true;
                    }
                }
            } else {
                break;
            }
        }
        
        return safe_paths;
    }

private:
    Path find_augmenting_path(VertexId source, VertexId sink, const std::vector<bool>& blocked) {
        // Modified BFS/SPFA respecting blocked vertices
        std::vector<Weight> dist(graph_.num_vertices(), INF_WEIGHT);
        std::vector<VertexId> parent(graph_.num_vertices(), NULL_VERTEX);
        std::vector<bool> in_queue(graph_.num_vertices(), false);
        std::queue<VertexId> q;
        
        dist[source] = 0;
        q.push(source);
        in_queue[source] = true;
        
        while (!q.empty()) {
            VertexId u = q.front(); q.pop();
            in_queue[u] = false;
            
            if (blocked[u] && u != source) continue;
            
            auto [neighbors, weights] = graph_.edges(u);
            for (size_t i = 0; i < neighbors.size(); ++i) {
                VertexId v = neighbors[i];
                if (blocked[v] && v != sink) continue;
                
                Weight w = weights[i];
                // Check capacity constraint
                EdgeId edge_idx = find_edge_index(u, v);
                if (edge_idx < 0 || dist[u] + w >= dist[v]) continue;
                
                dist[v] = dist[u] + w;
                parent[v] = u;
                
                if (!in_queue[v]) {
                    q.push(v);
                    in_queue[v] = true;
                }
            }
        }
        
        Path path;
        if (parent[sink] == NULL_VERTEX) return path;
        
        // Reconstruct path
        VertexId cur = sink;
        while (cur != NULL_VERTEX) {
            path.vertices.push_back(cur);
            cur = parent[cur];
        }
        std::reverse(path.vertices.begin(), path.vertices.end());
        
        // Calculate path properties
        path.flow = dist[sink];
        path.min_capacity = calculate_min_capacity(path.vertices);
        
        return path;
    }
    
    Weight calculate_min_capacity(const std::vector<VertexId>& path) {
        Weight min_cap = INF_WEIGHT;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            EdgeId idx = find_edge_index(path[i], path[i+1]);
            if (idx >= 0) {
                min_cap = std::min(min_cap, capacities_[idx]);
            }
        }
        return min_cap;
    }
    
    bool is_path_safe(const Path& path) {
        return path.flow <= path.min_capacity;
    }
    
    EdgeId find_edge_index(VertexId u, VertexId v) {
        // Binary search in CSR structure (neighbors are sorted)
        auto neighbors = graph_.neighbors(u);
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        if (it != neighbors.end() && *it == v) {
            return std::distance(neighbors.begin(), it) + graph_.offsets_data()[u];
        }
        return -1;
    }
};

} // namespace graph