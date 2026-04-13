# tests/CMakeLists.txt
add_executable(graph_tests
    test_main.cpp
    test_csr_graph.cpp
    test_spfa.cpp
    test_cycle_detection.cpp
    test_concurrent_queue.cpp
    test_safe_flow.cpp
)

target_link_libraries(graph_tests PRIVATE graph_engine GTest::gtest)

include(GoogleTest)
gtest_discover_tests(graph_tests)