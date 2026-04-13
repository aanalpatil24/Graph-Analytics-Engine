// tests/test_csr_graph.cpp
#include <gtest/gtest.h>
#include "graph/csr_graph.hpp"
#include "graph/graph_builder.hpp"

using namespace graph;

TEST(CSRGraphTest, BasicConstruction) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 5);
    builder.add_edge(0, 2, 3);
    builder.add_edge(1, 3, 6);
    builder.add_edge(1, 2, 2);
    builder.add_edge(2, 4, 4);
    builder.add_edge(2, 5, 2);
    builder.add_edge(2, 3, 7);
    builder.add_edge(3, 4, -1);
    builder.add_edge(3, 5, 1);
    builder.add_edge(4, 5, -2);
    
    auto graph = builder.build();
    
    EXPECT_EQ(graph.num_vertices(), 5);
    EXPECT_EQ(graph.num_edges(), 10);
    
    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.aligned_offsets()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.aligned_destinations()) % 64, 0);
}

TEST(CSRGraphTest, MoveSemantics) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 10);
    auto graph1 = builder.build();
    
    auto offsets = const_cast<uint32_t*>(graph1.aligned_offsets()); // Just checking address
    
    CSRGraph graph2 = std::move(graph1);
    
    EXPECT_EQ(graph2.num_edges(), 1);
    EXPECT_EQ(graph2.aligned_offsets(), offsets); // Same memory
}

TEST(CSRGraphTest, MemorySafety) {
    // This test designed for Valgrind memcheck
    {
        GraphBuilder builder(1000, 10000);
        for (int i = 0; i < 999; ++i) {
            builder.add_edge(i, i+1, i);
        }
        auto graph = builder.build();
        
        // Access all data
        volatile int sum = 0;
        for (size_t i = 0; i < graph.num_vertices(); ++i) {
            auto neighbors = graph.neighbors(i);
            for (auto n : neighbors) {
                sum += n;
            }
        }
    }
    // Valgrind should report no leaks here
}