// tests/test_safe_flow.cpp
#include <gtest/gtest.h>
#include "graphs/graph_builder.hpp"
#include "graphs/safe_flow.hpp"

using namespace graph;

TEST(SafeFlowTest, BasicPathFinding) {
    GraphBuilder builder(3, 2);
    // Simple path: 0 -> 1 -> 2 with weights of 1
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    
    auto graph = builder.build();
    std::vector<int32_t> caps = {10, 5}; // Bottleneck is 5
    
    SafeFlowDecomposition safe_flow(graph, caps);
    auto paths = safe_flow.find_safe_paths(0, 2);
    
    ASSERT_EQ(paths.size(), 1);
    EXPECT_EQ(paths[0].vertices.size(), 3);
    EXPECT_EQ(paths[0].min_capacity, 5);
}

TEST(SafeFlowTest, MultipleDisjointPaths) {
    GraphBuilder builder(4, 4);
    // Diamond graph: 0->1->3 and 0->2->3
    builder.add_edge(0, 1, 1);
    builder.add_edge(0, 2, 1);
    builder.add_edge(1, 3, 1);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.build();
    std::vector<int32_t> caps = {10, 10, 5, 10}; 
    
    SafeFlowDecomposition safe_flow(graph, caps);
    auto paths = safe_flow.find_safe_paths(0, 3);
    
    // Should find both the upper and lower paths
    ASSERT_EQ(paths.size(), 2);
    
    int32_t total_flow = paths[0].min_capacity + paths[1].min_capacity;
    EXPECT_EQ(total_flow, 15); // 10 flow through top, 5 flow through bottom
}