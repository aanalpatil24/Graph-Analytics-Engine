// src/graph_builder.cpp
#include "graph/graph_builder.hpp"
#include <algorithm>

namespace graph {

GraphBuilder::GraphBuilder(size_t expected_vertices, size_t expected_edges) {
    adjacency_.reserve(expected_vertices);
    // Reserve edges roughly per vertex
    for (size_t i = 0; i < expected_vertices; ++i) {
        adjacency_.emplace_back();
        adjacency_.back().reserve(expected_edges / expected_vertices);
    }
}

void GraphBuilder::add_vertex() {
    adjacency_.emplace_back();
}

void GraphBuilder::add_edge(uint32_t from, uint32_t to, int32_t weight) {
    if (from >= adjacency_.size()) {
        adjacency_.resize(from + 1);
    }
    adjacency_[from].emplace_back(to, weight);
    ++total_edges_;
}

CSRGraph GraphBuilder::build() {
    const size_t n = adjacency_.size();
    const size_t m = total_edges_;
    
    CSRGraph::AlignedUint32Vec offsets(n + 1);
    CSRGraph::AlignedEdgeDestVec destinations;
    CSRGraph::AlignedInt32Vec weights;
    
    destinations.reserve(m);
    weights.reserve(m);
    
    // Compute offsets
    offsets[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + adjacency_[i].size();
    }
    
    // Ensure alignment by padding if necessary (shouldn't happen with aligned_alloc)
    while (reinterpret_cast<uintptr_t>(offsets.data()) % 64 != 0) {
        // This shouldn't happen with AlignedAllocator, but safety check
    }
    
    // Flatten edges
    for (auto& vec : adjacency_) {
        // Sort by destination for cache locality
        std::sort(vec.begin(), vec.end(), 
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        for (auto& [dest, weight] : vec) {
            destinations.push_back(dest);
            weights.push_back(weight);
        }
    }
    
    // Clear builder state (move semantics)
    adjacency_.clear();
    total_edges_ = 0;
    
    return CSRGraph(std::move(offsets), std::move(destinations), 
                    std::move(weights), n);
}

void GraphBuilder::clear() {
    adjacency_.clear();
    total_edges_ = 0;
}

}