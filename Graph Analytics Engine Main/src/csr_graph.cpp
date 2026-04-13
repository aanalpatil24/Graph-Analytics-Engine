// src/csr_graph.cpp
#include "graph/csr_graph.hpp"

namespace graph {

CSRGraph::CSRGraph(AlignedUint32Vec offsets, AlignedEdgeDestVec destinations,
                   AlignedInt32Vec weights, size_t num_vertices)
    : offsets_(std::move(offsets))
    , destinations_(std::move(destinations))
    , weights_(std::move(weights))
    , num_vertices_(num_vertices) {
    
    // Validate alignment for AVX-512
    if (reinterpret_cast<uintptr_t>(offsets_.data()) % 64 != 0 ||
        reinterpret_cast<uintptr_t>(destinations_.data()) % 64 != 0 ||
        reinterpret_cast<uintptr_t>(weights_.data()) % 64 != 0) {
        throw std::runtime_error("CSRGraph arrays not 64-byte aligned");
    }
}

}