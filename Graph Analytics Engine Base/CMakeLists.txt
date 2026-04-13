# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(GraphAnalyticsEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -mtune=native)
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    # AVX-512 flags
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        add_compile_options(-mavx512f)
        add_definitions(-DHAS_AVX512)
    endif()
    
    # Sanitizers for debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-fsanitize=address,undefined -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
elseif(MSVC)
    add_compile_options(/O2 /W4 /EHsc)
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        add_compile_options(/arch:AVX512)
        add_definitions(-DHAS_AVX512)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Main library (header-only, but create interface)
add_library(graph_engine INTERFACE)
target_include_directories(graph_engine INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Testing
enable_testing()
add_executable(graph_engine_tests
    tests/test_main.cpp
    tests/test_csr_graph.cpp
    tests/test_spfa.cpp
    tests/test_concurrent_toposort.cpp
    tests/test_safe_flow.cpp
)

target_link_libraries(graph_engine_tests 
    graph_engine
    GTest::gtest_main
    pthread
)

include(GoogleTest)
gtest_discover_tests(graph_engine_tests)

# Benchmark executable
add_executable(benchmark_spfa benchmarks/benchmark_spfa.cpp)
target_link_libraries(benchmark_spfa graph_engine pthread)

# Installation
install(DIRECTORY include/ DESTINATION include)