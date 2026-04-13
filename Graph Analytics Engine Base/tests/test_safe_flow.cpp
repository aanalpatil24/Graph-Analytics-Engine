// tests/test_safe_flow.cpp
#include <gtest/gtest.h>
#include "graph/graph_builder.hpp"
#include "algorithms/safe_flow.hpp"

using namespace graph;

TEST(SafeFlowTest, BasicPathFinding) {
    GraphBuilder builder;
    // Simple path: 0 -> 1 -> 2 with capacities [10, 5]
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    
    auto graph = builder.finish();
    std::vector<Weight> caps = {10, 5};
    
    SafeFlowDecomposition safe_flow(graph, caps);
    auto paths = safe_flow.find_safe_paths(0, 2);
    
    ASSERT_EQ(paths.size(), 1);
    EXPECT_EQ(paths[0].vertices.size(), 3);
    EXPECT_EQ(paths[0].min_capacity, 5);
}

TEST(SafeFlowTest, MultiplePaths) {
    GraphBuilder builder;
    // Diamond graph with two paths 0->1->3 and 0->2->3
    builder.add_edge(0, 1, 1);
    builder.add_edge(0, 2, 2);
    builder.add_edge(1, 3, 1);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.finish();
    std::vector<Weight> caps(graph.num_edges(), 10);
    
    SafeFlowDecomposition safe_flow(graph, caps);
    auto paths = safe_flow.find_safe_paths(0, 3);
    
    EXPECT_GE(paths.size(), 1);
}