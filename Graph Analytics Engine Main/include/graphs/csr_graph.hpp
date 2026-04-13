// include/graph/csr_graph.hpp
#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <span>
#include "aligned_allocator.hpp"

namespace graph {

struct Edge {
    uint32_t destination;
    int32_t weight;
};

class CSRGraph {
public:
    using IndexType = uint32_t;
    using WeightType = int32_t;
    using OffsetType = uint32_t;
    
    using AlignedUint32Vec = std::vector<uint32_t, memory::AlignedAllocator<uint32_t, 64>>;
    using AlignedInt32Vec = std::vector<int32_t, memory::AlignedAllocator<int32_t, 64>>;
    using AlignedEdgeDestVec = std::vector<uint32_t, memory::AlignedAllocator<uint32_t, 64>>;

    CSRGraph() = default;
    
    // Move-only semantics for zero-copy transfer
    CSRGraph(CSRGraph&&) noexcept = default;
    CSRGraph& operator=(CSRGraph&&) noexcept = default;
    
    // Disable copy to prevent accidental expensive copies
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;

    // Construct from builder components
    CSRGraph(AlignedUint32Vec offsets, AlignedEdgeDestVec destinations, 
             AlignedInt32Vec weights, size_t num_vertices);

    [[nodiscard]] size_t num_vertices() const noexcept { return num_vertices_; }
    [[nodiscard]] size_t num_edges() const noexcept { return destinations_.size(); }
    
    [[nodiscard]] std::span<const uint32_t> offsets() const noexcept {
        return std::span{offsets_.data(), offsets_.size()};
    }
    
    [[nodiscard]] std::span<const uint32_t> destinations() const noexcept {
        return std::span{destinations_.data(), destinations_.size()};
    }
    
    [[nodiscard]] std::span<const int32_t> weights() const noexcept {
        return std::span{weights_.data(), weights_.size()};
    }

    [[nodiscard]] std::span<const uint32_t> neighbors(uint32_t vertex) const noexcept {
        const auto start = offsets_[vertex];
        const auto end = offsets_[vertex + 1];
        return std::span{destinations_.data() + start, end - start};
    }

    [[nodiscard]] std::span<const int32_t> neighbor_weights(uint32_t vertex) const noexcept {
        const auto start = offsets_[vertex];
        const auto end = offsets_[vertex + 1];
        return std::span{weights_.data() + start, end - start};
    }

    // Cache-line aligned access for AVX-512
    [[nodiscard]] const uint32_t* aligned_offsets() const noexcept { return offsets_.data(); }
    [[nodiscard]] const uint32_t* aligned_destinations() const noexcept { return destinations_.data(); }
    [[nodiscard]] const int32_t* aligned_weights() const noexcept { return weights_.data(); }

private:
    AlignedUint32Vec offsets_;        // Size: num_vertices + 1, 64-byte aligned
    AlignedEdgeDestVec destinations_; // Size: num_edges, 64-byte aligned  
    AlignedInt32Vec weights_;         // Size: num_edges, 64-byte aligned
    size_t num_vertices_{0};
};

}