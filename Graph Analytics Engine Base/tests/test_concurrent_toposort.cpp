// tests/test_concurrent_toposort.cpp
#include <gtest/gtest.h>
#include "graph/graph_builder.hpp"
#include "algorithms/concurrent_toposort.hpp"
#include "utils/thread_pool.hpp"

using namespace graph;

TEST(TopoSortTest, AcyclicGraph) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 1);
    builder.add_edge(0, 2, 1);
    builder.add_edge(1, 3, 1);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.finish();
    ThreadPool pool(4);
    ConcurrentTopologicalSort toposort(graph);
    
    auto result = toposort.compute(pool);
    
    EXPECT_FALSE(result.has_cycle);
    EXPECT_EQ(result.order.size(), 4);
    
    // Verify topological order property
    std::vector<size_t> pos(4);
    for (size_t i = 0; i < result.order.size(); ++i) {
        pos[result.order[i]] = i;
    }
    EXPECT_LT(pos[0], pos[1]);
    EXPECT_LT(pos[0], pos[2]);
    EXPECT_LT(pos[1], pos[3]);
    EXPECT_LT(pos[2], pos[3]);
}

TEST(TopoSortTest, CycleDetection) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    builder.add_edge(2, 0, 1); // Cycle: 0->1->2->0
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.finish();
    ThreadPool pool(4);
    ConcurrentTopologicalSort toposort(graph);
    
    auto result = toposort.compute(pool);
    
    EXPECT_TRUE(result.has_cycle);
    EXPECT_FALSE(result.cycles.empty());
    
    // Verify cycle contains vertices 0, 1, 2
    bool found_cycle = false;
    for (const auto& cycle : result.cycles) {
        if (cycle.size() == 3) {
            std::vector<bool> has(4, false);
            for (VertexId v : cycle) has[v] = true;
            if (has[0] && has[1] && has[2]) found_cycle = true;
        }
    }
    EXPECT_TRUE(found_cycle);
}