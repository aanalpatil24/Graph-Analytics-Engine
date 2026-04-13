# benchmarks/CMakeLists.txt

# Original SPFA Engine Benchmark
add_executable(benchmark_spfa benchmark_spfa.cpp)
target_link_libraries(benchmark_spfa PRIVATE graph_engine)


# NEW: Safe Flow Decomposition Benchmark
add_executable(benchmark_safe_flow benchmark_safe_flow.cpp)
target_link_libraries(benchmark_safe_flow PRIVATE graph_engine)
