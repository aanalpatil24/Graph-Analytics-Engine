// tests/test_spfa.cpp
#include <gtest/gtest.h>
#include "graph/csr_graph.hpp"
#include "graph/graph_builder.hpp"
#include "graph/spfa_engine.hpp"
#include "graph/thread_pool.hpp"
#include <limits>

using namespace graph;

class SPFATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a graph with negative weights but no negative cycles
        GraphBuilder builder;
        builder.add_edge(0, 1, 6);
        builder.add_edge(0, 2, 7);
        builder.add_edge(1, 2, 8);
        builder.add_edge(1, 3, 5);
        builder.add_edge(1, 4, -4);
        builder.add_edge(2, 3, -3);
        builder.add_edge(2, 4, 9);
        builder.add_edge(3, 1, -2);
        builder.add_edge(4, 0, 2);
        builder.add_edge(4, 3, 7);
        
        graph_ = std::make_unique<CSRGraph>(builder.build());
        pool_ = std::make_unique<ThreadPool>(4);
    }
    
    std::unique_ptr<CSRGraph> graph_;
    std::unique_ptr<ThreadPool> pool_;
};

TEST_F(SPFATest, ScalarCorrectness) {
    SPFASolver solver(*graph_, *pool_);
    auto result = solver.solve_scalar(0);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.distances[0].load(), 0);
    EXPECT_EQ(result.distances[1].load(), 2);  // 0->2->3->1 = 7-3-2 = 2
    EXPECT_EQ(result.distances[2].load(), 7);
    EXPECT_EQ(result.distances[3].load(), 4);  // 0->2->3 = 7-3 = 4
    EXPECT_EQ(result.distances[4].load(), -2); // 0->1->4 = 6-4 = 2... wait
    
    // Actually: 0->1 (6) ->4 (-4) = 2, or 0->2 (7) ->3 (-3) ->1 (-2) ->4 (-4) = -2
    EXPECT_EQ(result.distances[4].load(), -2);
}

TEST_F(SPFATest, VectorizedCorrectness) {
    SPFASolver solver(*graph_, *pool_);
    auto result_vec = solver.solve_vectorized(0);
    auto result_scalar = solver.solve_scalar(0);
    
    EXPECT_TRUE(result_vec.success);
    
    for (size_t i = 0; i < graph_->num_vertices(); ++i) {
        EXPECT_EQ(result_vec.distances[i].load(), result_scalar.distances[i].load())
            << "Mismatch at vertex " << i;
    }
}

TEST_F(SPFATest, ConcurrentCorrectness) {
    SPFASolver solver(*graph_, *pool_);
    auto result_conc = solver.solve_concurrent(0, 4);
    auto result_scalar = solver.solve_scalar(0);
    
    EXPECT_TRUE(result_conc.success);
    
    for (size_t i = 0; i < graph_->num_vertices(); ++i) {
        EXPECT_EQ(result_conc.distances[i].load(), result_scalar.distances[i].load())
            << "Mismatch at vertex " << i;
    }
}

TEST_F(SPFATest, LargeGraphPerformance) {
    // Generate larger graph for performance testing
    const int N = 1000;
    GraphBuilder builder;
    
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= 10 && i + j < N; ++j) {
            builder.add_edge(i, i + j, j * 2 - 1);
        }
    }
    
    auto large_graph = builder.build();
    SPFASolver solver(large_graph, *pool_);
    
    // Time the vectorized version
    auto start = std::chrono::high_resolution_clock::now();
    auto result = solver.solve_vectorized(0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Vectorized SPFA on " << N << " nodes took " 
              << duration.count() << " us\n";
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.distances[0].load(), 0);
}