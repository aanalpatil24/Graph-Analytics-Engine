// include/graph/graph_builder.hpp
#pragma once
#include <vector>
#include <utility>
#include "csr_graph.hpp"

namespace graph {

class GraphBuilder {
public:
    explicit GraphBuilder(size_t expected_vertices = 0, size_t expected_edges = 0);
    
    void add_edge(uint32_t from, uint32_t to, int32_t weight);
    void add_vertex();
    
    [[nodiscard]] size_t current_vertices() const noexcept { return adjacency_.size(); }
    [[nodiscard]] size_t current_edges() const noexcept { return total_edges_; }

    // Convert to CSR with move semantics - zero copy
    [[nodiscard]] CSRGraph build();

    void clear();

private:
    std::vector<std::vector<std::pair<uint32_t, int32_t>>> adjacency_;
    size_t total_edges_{0};
};

}