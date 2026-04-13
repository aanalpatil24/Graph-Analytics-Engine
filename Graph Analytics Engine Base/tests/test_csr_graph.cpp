// tests/test_csr_graph.cpp
#include <gtest/gtest.h>
#include "graph/graph_builder.hpp"
#include "graph/csr_graph.hpp"

using namespace graph;

TEST(CSRGraphTest, BasicConstruction) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 10);
    builder.add_edge(0, 2, 5);
    builder.add_edge(1, 2, 2);
    builder.add_edge(1, 3, 1);
    builder.add_edge(2, 3, 9);
    
    auto graph = builder.finish();
    
    EXPECT_EQ(graph.num_vertices(), 4);
    EXPECT_EQ(graph.num_edges(), 5);
    
    auto neighbors = graph.neighbors(0);
    EXPECT_EQ(neighbors.size(), 2);
    EXPECT_EQ(neighbors[0], 1);
    EXPECT_EQ(neighbors[1], 2);
}

TEST(CSRGraphTest, MoveSemantics) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 5);
    auto graph1 = builder.finish();
    
    auto graph2 = std::move(graph1);
    EXPECT_EQ(graph2.num_vertices(), 2);
    EXPECT_EQ(graph1.num_vertices(), 0); // Moved-from state
}

TEST(CSRGraphTest, CacheAlignment) {
    GraphBuilder builder;
    for (int i = 0; i < 100; ++i) {
        builder.add_edge(i, (i+1) % 100, i);
    }
    auto graph = builder.finish();
    
    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.offsets_data()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.destinations_data()) % 64, 0);
}