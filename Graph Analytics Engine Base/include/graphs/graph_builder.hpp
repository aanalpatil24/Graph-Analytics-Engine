// include/graph/graph_builder.hpp
#pragma once
#include <vector>
#include <algorithm>
#include "graph/csr_graph.hpp"

namespace graph {

/**
 * Incremental graph builder that constructs CSR representation
 * Uses move semantics for zero-copy transfer to immutable CSRGraph
 */
class GraphBuilder {
private:
    struct EdgeEntry {
        VertexId src;
        VertexId dst;
        Weight weight;
        
        bool operator<(const EdgeEntry& other) const {
            return src < other.src || (src == other.src && dst < other.dst);
        }
    };

    std::vector<EdgeEntry> edges_;
    size_t num_vertices_ = 0;

public:
    void add_edge(VertexId src, VertexId dst, Weight weight) {
        edges_.push_back({src, dst, weight});
        num_vertices_ = std::max(num_vertices_, static_cast<size_t>(std::max(src, dst) + 1));
    }

    void reserve_edges(size_t n) { edges_.reserve(n); }
    void reserve_vertices(size_t n) { num_vertices_ = std::max(num_vertices_, n); }

    /**
     * Finalize construction and return immutable CSRGraph
     * O(E log E) sorting complexity for CSR construction
     */
    CSRGraph finish() {
        // Sort by source vertex for CSR construction
        std::sort(edges_.begin(), edges_.end());
        
        CacheAlignedVector<EdgeId> offsets(num_vertices_ + 1, 0);
        CacheAlignedVector<VertexId> destinations;
        CacheAlignedVector<Weight> weights;
        
        destinations.reserve(edges_.size());
        weights.reserve(edges_.size());
        
        // Count edges per vertex
        for (const auto& e : edges_) {
            offsets[e.src + 1]++;
        }
        
        // Prefix sum for offsets
        for (size_t i = 1; i <= num_vertices_; ++i) {
            offsets[i] += offsets[i - 1];
        }
        
        // Fill destinations and weights
        destinations.resize(edges_.size());
        weights.resize(edges_.size());
        
        std::vector<EdgeId> current_offset(offsets.begin(), offsets.end() - 1);
        
        for (const auto& e : edges_) {
            EdgeId pos = current_offset[e.src]++;
            destinations[pos] = e.dst;
            weights[pos] = e.weight;
        }
        
        // Clear builder state to free memory
        edges_.clear();
        edges_.shrink_to_fit();
        
        return CSRGraph(std::move(offsets), std::move(destinations), std::move(weights));
    }
};

} // namespace graph