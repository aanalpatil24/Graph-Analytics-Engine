# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(GraphAnalyticsEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for performance and safety
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -mavx512f -mavx512vl -march=native -mtune=native)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-fstrict-aliasing -funroll-loops)
    
    # Debug/Profile builds
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_PROFILE "-g -O2 -DENABLE_PROFILING")
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX512 /W4 /WX)
endif()

# Find packages
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Library
add_library(graph_engine STATIC
    src/csr_graph.cpp
    src/graph_builder.cpp
    src/spfa_engine.cpp
    src/cycle_detector.cpp
    src/thread_pool.cpp
)

target_include_directories(graph_engine PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(graph_engine PUBLIC Threads::Threads)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# The graph and baseline headers.
target_include_directories(graph_engine PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)