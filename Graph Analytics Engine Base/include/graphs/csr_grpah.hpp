// include/graph/csr_graph.hpp
#pragma once
#include <vector>
#include <span>
#include <cassert>
#include "graph/types.hpp"
#include "utils/aligned_allocator.hpp"

namespace graph {

/**
 * Immutable CSR Graph representation with 64-byte aligned memory
 * Zero-copy move semantics for optimal performance
 */
class CSRGraph {
public:
    struct Edge {
        VertexId dst;
        Weight weight;
    };

private:
    // Cache-aligned contiguous arrays
    CacheAlignedVector<EdgeId> vertex_offsets_;      // Size: num_vertices + 1
    CacheAlignedVector<VertexId> edge_destinations_; // Size: num_edges
    CacheAlignedVector<Weight> edge_weights_;        // Size: num_edges
    
    size_t num_vertices_ = 0;
    size_t num_edges_ = 0;

public:
    CSRGraph() = default;
    
    // Move-only semantics for zero-copy transfer
    CSRGraph(CSRGraph&& other) noexcept 
        : vertex_offsets_(std::move(other.vertex_offsets_))
        , edge_destinations_(std::move(other.edge_destinations_))
        , edge_weights_(std::move(other.edge_weights_))
        , num_vertices_(other.num_vertices_)
        , num_edges_(other.num_edges_) {
        other.num_vertices_ = 0;
        other.num_edges_ = 0;
    }

    CSRGraph& operator=(CSRGraph&& other) noexcept {
        if (this != &other) {
            vertex_offsets_ = std::move(other.vertex_offsets_);
            edge_destinations_ = std::move(other.edge_destinations_);
            edge_weights_ = std::move(other.edge_weights_);
            num_vertices_ = other.num_vertices_;
            num_edges_ = other.num_edges_;
            other.num_vertices_ = 0;
            other.num_edges_ = 0;
        }
        return *this;
    }

    // Delete copy to enforce move semantics
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;

    // Constructors for builder
    CSRGraph(CacheAlignedVector<EdgeId> offsets,
             CacheAlignedVector<VertexId> dsts,
             CacheAlignedVector<Weight> weights)
        : vertex_offsets_(std::move(offsets))
        , edge_destinations_(std::move(dsts))
        , edge_weights_(std::move(weights))
        , num_vertices_(vertex_offsets_.size() - 1)
        , num_edges_(edge_destinations_.size()) {}

    // Accessors with bounds checking in debug mode
    [[nodiscard]] std::span<const VertexId> neighbors(VertexId v) const noexcept {
        assert(v >= 0 && static_cast<size_t>(v) < num_vertices_);
        const auto start = vertex_offsets_[v];
        const auto end = vertex_offsets_[v + 1];
        return std::span(edge_destinations_.data() + start, end - start);
    }

    [[nodiscard]] std::span<const Weight> weights(VertexId v) const noexcept {
        assert(v >= 0 && static_cast<size_t>(v) < num_vertices_);
        const auto start = vertex_offsets_[v];
        const auto end = vertex_offsets_[v + 1];
        return std::span(edge_weights_.data() + start, end - start);
    }

    [[nodiscard]] std::pair<std::span<const VertexId>, std::span<const Weight>> 
    edges(VertexId v) const noexcept {
        assert(v >= 0 && static_cast<size_t>(v) < num_vertices_);
        const auto start = vertex_offsets_[v];
        const auto end = vertex_offsets_[v + 1];
        return {
            std::span(edge_destinations_.data() + start, end - start),
            std::span(edge_weights_.data() + start, end - start)
        };
    }

    [[nodiscard]] size_t num_vertices() const noexcept { return num_vertices_; }
    [[nodiscard]] size_t num_edges() const noexcept { return num_edges_; }
    
    // Raw data access for SIMD algorithms
    [[nodiscard]] const EdgeId* offsets_data() const noexcept { return vertex_offsets_.data(); }
    [[nodiscard]] const VertexId* destinations_data() const noexcept { return edge_destinations_.data(); }
    [[nodiscard]] const Weight* weights_data() const noexcept { return edge_weights_.data(); }
};

} // namespace graph