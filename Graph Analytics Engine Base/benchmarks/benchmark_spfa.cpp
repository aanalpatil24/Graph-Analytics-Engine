// benchmarks/benchmark_spfa.cpp
#include <iostream>
#include <chrono>
#include <random>
#include "graph/graph_builder.hpp"
#include "algorithms/spfa_avx512.hpp"

using namespace graph;

void benchmark_random_graph(size_t n, size_t m) {
    GraphBuilder builder;
    builder.reserve_vertices(n);
    builder.reserve_edges(m);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<VertexId> vertex_dist(0, n-1);
    std::uniform_int_distribution<Weight> weight_dist(1, 100);
    
    for (size_t i = 0; i < m; ++i) {
        builder.add_edge(vertex_dist(rng), vertex_dist(rng), weight_dist(rng));
    }
    
    auto graph = builder.finish();
    
    auto start = std::chrono::high_resolution_clock::now();
    AVX512SPFA spfa(graph);
    auto dist = spfa.compute_shortest_paths(0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Graph: " << n << " vertices, " << m << " edges\n";
    std::cout << "Time: " << ms << "ms\n";
    std::cout << "Throughput: " << (m / (ms / 1000.0)) / 1e6 << " M edges/sec\n\n";
}

int main() {
    std::cout << "=== Graph Analytics Engine Benchmark ===\n\n";
    
    std::cout << "Small graph (1K vertices, 10K edges):\n";
    benchmark_random_graph(1000, 10000);
    
    std::cout << "Medium graph (10K vertices, 100K edges):\n";
    benchmark_random_graph(10000, 100000);
    
    std::cout << "Large graph (100K vertices, 1M edges):\n";
    benchmark_random_graph(100000, 1000000);
    
    return 0;
}