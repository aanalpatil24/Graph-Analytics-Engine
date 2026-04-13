// tests/test_cycle_detection.cpp
#include <gtest/gtest.h>
#include "graph/csr_graph.hpp"
#include "graph/graph_builder.hpp"
#include "graph/cycle_detector.hpp"
#include "graph/thread_pool.hpp"

using namespace graph;

TEST(CycleDetectionTest, NoCycleDAG) {
    GraphBuilder builder;
    // DAG: 0 -> 1 -> 2 -> 3
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    CycleDetector detector(graph, pool);
    
    auto info = detector.detect_and_resolve();
    
    EXPECT_FALSE(info.has_cycle);
    EXPECT_EQ(info.topological_order.size(), 4);
    EXPECT_TRUE(info.cycle_nodes.empty());
}

TEST(CycleDetectionTest, SimpleCycle) {
    GraphBuilder builder;
    // Cycle: 0 -> 1 -> 2 -> 0
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    builder.add_edge(2, 0, 1);
    // Additional edge to 3
    builder.add_edge(0, 3, 1);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    CycleDetector detector(graph, pool);
    
    auto info = detector.detect_and_resolve();
    
    EXPECT_TRUE(info.has_cycle);
    EXPECT_EQ(info.topological_order.size(), 1); // Only node 3
    EXPECT_EQ(info.cycle_nodes.size(), 3); // 0, 1, 2
}

TEST(CycleDetectionTest, ComplexGraphWithCycle) {
    GraphBuilder builder;
    // DAG part
    builder.add_edge(0, 1, 1);
    builder.add_edge(0, 2, 1);
    builder.add_edge(1, 3, 1);
    builder.add_edge(2, 3, 1);
    
    // Cycle part
    builder.add_edge(4, 5, 1);
    builder.add_edge(5, 6, 1);
    builder.add_edge(6, 4, 1);
    
    // Connect them
    builder.add_edge(3, 4, 1);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    CycleDetector detector(graph, pool);
    
    auto info = detector.detect_and_resolve();
    
    EXPECT_TRUE(info.has_cycle);
    EXPECT_EQ(info.topological_order.size(), 4); // 0, 1, 2, 3
    EXPECT_EQ(info.cycle_nodes.size(), 3); // 4, 5, 6
}