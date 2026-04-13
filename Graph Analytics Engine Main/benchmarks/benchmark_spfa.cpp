// benchmarks/benchmark_spfa.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "graph/csr_graph.hpp"
#include "graph/graph_builder.hpp"
#include "graph/spfa_engine.hpp"
#include "graph/thread_pool.hpp"

using namespace graph;

class Benchmark {
public:
    static void run() {
        std::cout << "Graph Analytics Engine Benchmark\n";
        std::cout << "================================\n\n";
        
        benchmark_sparse_graph(10000, 50000);
        benchmark_sparse_graph(100000, 500000);
        benchmark_dense_graph(1000);
        
        benchmark_cycle_detection(10000, 50000);
    }

private:
    static void benchmark_sparse_graph(int n, int m) {
        std::cout << "Sparse Graph: " << n << " vertices, " << m << " edges\n";
        
        auto graph = generate_random_graph(n, m);
        ThreadPool pool(std::thread::hardware_concurrency());
        SPFASolver solver(graph, pool);
        
        // Warmup
        solver.solve_scalar(0);
        
        // Scalar benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            solver.solve_scalar(0);
        }
        auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count() / 10.0;
        
        // Vectorized benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            solver.solve_vectorized(0);
        }
        auto vec_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count() / 10.0;
        
        // Concurrent benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            solver.solve_concurrent(0);
        }
        auto conc_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count() / 10.0;
        
        std::cout << "  Scalar:    " << scalar_time << " us\n";
        std::cout << "  Vectorized: " << vec_time << " us (speedup: " 
                  << scalar_time/vec_time << "x)\n";
        std::cout << "  Concurrent: " << conc_time << " us (speedup: " 
                  << scalar_time/conc_time << "x)\n\n";
    }
    
    static void benchmark_dense_graph(int n) {
        std::cout << "Dense Graph: " << n << " vertices, ~" << n*(n-1)/2 << " edges\n";
        
        GraphBuilder builder;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-10, 100);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                if (rng() % 2 == 0) {
                    builder.add_edge(i, j, dist(rng));
                }
            }
        }
        
        auto graph = builder.build();
        ThreadPool pool;
        SPFASolver solver(graph, pool);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = solver.solve_vectorized(0);
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "  Vectorized SPFA: " << time << " ms\n\n";
    }
    
    static void benchmark_cycle_detection(int n, int m) {
        std::cout << "Cycle Detection: " << n << " vertices, " << m << " edges\n";
        
        auto graph = generate_random_graph(n, m);
        ThreadPool pool;
        CycleDetector detector(graph, pool);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto info = detector.detect_and_resolve();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        std::cout << "  Time: " << time << " ms\n";
        std::cout << "  Has cycle: " << (info.has_cycle ? "yes" : "no") << "\n";
        std::cout << "  Topo order size: " << info.topological_order.size() << "\n\n";
    }
    
    static CSRGraph generate_random_graph(int n, int m) {
        GraphBuilder builder(n, m);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> vertex_dist(0, n-1);
        std::uniform_int_distribution<int> weight_dist(-5, 100);
        
        for (int i = 0; i < m; ++i) {
            int u = vertex_dist(rng);
            int v = vertex_dist(rng);
            if (u != v) {
                builder.add_edge(u, v, weight_dist(rng));
            }
        }
        
        return builder.build();
    }
};

int main() {
    Benchmark::run();
    return 0;
}