#pragma once
#include <vector>

namespace baseline {

struct Edge {
    int dst;
    int weight;
};

/**
 * @brief Standard Object-Oriented Graph Representation.
 * * BASELINE BOTTLENECK: Uses std::vector<std::vector<Edge>>. 
 * This dynamically allocates small blocks of memory across the heap. 
 * When the CPU traverses this graph, it suffers severe L1/L2 cache misses 
 * because the memory is not contiguous (unlike the optimized CSR format).
 */
class Graph {
public:
    std::vector<std::vector<Edge>> adj_list;
    int num_vertices;
    int num_edges;

    explicit Graph(int vertices) : num_vertices(vertices), num_edges(0) {
        adj_list.resize(vertices);
    }

    void add_edge(int src, int dst, int weight) {
        adj_list[src].push_back({dst, weight});
        num_edges++;
    }
};

} // namespace baseline