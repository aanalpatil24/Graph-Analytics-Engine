// tests/test_spfa.cpp
#include <gtest/gtest.h>
#include "graph/graph_builder.hpp"
#include "algorithms/spfa_avx512.hpp"
#include <chrono>

using namespace graph;

TEST(SPFATest, BasicShortestPath) {
    GraphBuilder builder;
    // Build a simple graph: 0 -> 1 (1), 0 -> 2 (4), 1 -> 2 (1), 1 -> 3 (3), 2 -> 3 (1)
    builder.add_edge(0, 1, 1);
    builder.add_edge(0, 2, 4);
    builder.add_edge(1, 2, 1);
    builder.add_edge(1, 3, 3);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.finish();
    AVX512SPFA spfa(graph);
    
    auto dist = spfa.compute_shortest_paths(0);
    
    EXPECT_EQ(dist[0], 0);
    EXPECT_EQ(dist[1], 1);
    EXPECT_EQ(dist[2], 2); // 0->1->2
    EXPECT_EQ(dist[3], 3); // 0->1->2->3
}

TEST(SPFATest, LargeGraphPerformance) {
    // Create dense graph to test AVX-512 vectorization
    GraphBuilder builder;
    const int N = 1000;
    
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < std::min(i+20, N); ++j) {
            builder.add_edge(i, j, (j - i) * 2);
        }
    }
    
    auto graph = builder.finish();
    AVX512SPFA spfa(graph);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto dist = spfa.compute_shortest_paths(0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_LT(duration.count(), 100000); // Should complete in <100ms
    
    EXPECT_EQ(dist[0], 0);
    EXPECT_EQ(dist[1], 2);
}