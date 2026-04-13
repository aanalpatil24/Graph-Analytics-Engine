#pragma once
#include <vector>

namespace baseline {

struct Edge {
    int dst;
    int weight;
};

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