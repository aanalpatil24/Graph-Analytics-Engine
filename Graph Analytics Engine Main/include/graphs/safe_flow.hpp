// include/graph/safe_flow.hpp
#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <queue>
#include "graph/csr_graph.hpp"

namespace graph {

/**
 * Safe Flow Decomposition - Identifies maximal safe paths in Directed Flow Networks.
 * Operates on top of the immutable CSRGraph, maintaining its own residual capacities.
 */
class SafeFlowDecomposition {
private:
    const CSRGraph& graph_;
    std::vector<int32_t> residual_capacities_;
    
public:
    struct Path {
        std::vector<uint32_t> vertices;
        int32_t min_capacity;
    };

    explicit SafeFlowDecomposition(const CSRGraph& graph, std::vector<int32_t> initial_capacities)
        : graph_(graph), residual_capacities_(std::move(initial_capacities)) {}

    std::vector<Path> find_safe_paths(uint32_t source, uint32_t sink) {
        std::vector<Path> safe_paths;
        
        while (true) {
            Path path = find_capacity_aware_path(source, sink);
            
            if (path.vertices.empty() || path.min_capacity <= 0) {
                break; // No more safe paths available
            }
            
            safe_paths.push_back(path);
            update_residual_network(path);
        }
        
        return safe_paths;
    }

private:
    // Capacity-aware BFS to find valid augmenting paths
    Path find_capacity_aware_path(uint32_t source, uint32_t sink) const {
        const size_t n = graph_.num_vertices();
        std::vector<uint32_t> parent(n, std::numeric_limits<uint32_t>::max());
        std::queue<uint32_t> q;
        
        q.push(source);
        parent[source] = source; // Mark as visited
        
        const auto* offsets = graph_.aligned_offsets();
        const auto* dests = graph_.aligned_destinations();
        
        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();
            
            if (u == sink) break;
            
            uint32_t start = offsets[u];
            uint32_t end = offsets[u + 1];
            
            for (uint32_t i = start; i < end; ++i) {
                uint32_t v = dests[i];
                
                // Only traverse if the edge has remaining capacity and vertex is unvisited
                if (residual_capacities_[i] > 0 && parent[v] == std::numeric_limits<uint32_t>::max()) {
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        
        Path path;
        path.min_capacity = 0;
        
        if (parent[sink] == std::numeric_limits<uint32_t>::max()) {
            return path; // No path found
        }
        
        // Reconstruct path
        uint32_t current = sink;
        path.min_capacity = std::numeric_limits<int32_t>::max();
        
        while (current != source) {
            path.vertices.push_back(current);
            uint32_t p = parent[current];
            
            // Find edge index from p to current
            uint32_t edge_idx = find_edge_index(p, current);
            path.min_capacity = std::min(path.min_capacity, residual_capacities_[edge_idx]);
            
            current = p;
        }
        path.vertices.push_back(source);
        std::reverse(path.vertices.begin(), path.vertices.end());
        
        return path;
    }
    
    void update_residual_network(const Path& path) {
        for (size_t i = 0; i + 1 < path.vertices.size(); ++i) {
            uint32_t u = path.vertices[i];
            uint32_t v = path.vertices[i+1];
            uint32_t edge_idx = find_edge_index(u, v);
            
            // Deduct the safe flow from the residual capacity
            residual_capacities_[edge_idx] -= path.min_capacity;
        }
    }
    
    uint32_t find_edge_index(uint32_t u, uint32_t v) const {
        const auto* offsets = graph_.aligned_offsets();
        const auto* dests = graph_.aligned_destinations();
        
        uint32_t start = offsets[u];
        uint32_t end = offsets[u + 1];
        
        for (uint32_t i = start; i < end; ++i) {
            if (dests[i] == v) return i;
        }
        return std::numeric_limits<uint32_t>::max();
    }
};

} // namespace graph