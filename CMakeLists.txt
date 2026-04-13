# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(GraphAnalyticsEngine VERSION 1.0.0 LANGUAGES CXX)

# Global C++20 Enforcement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and Warning Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -mtune=native)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    
    # Check and enable AVX-512 support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        add_compile_options(-mavx512f)
        add_definitions(-DHAS_AVX512)
    endif()
    
    # Sanitizers for Debug mode (Matches your CI/CD workflow)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-fsanitize=address,undefined -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Fetch Google Test globally for both Baseline and Optimized tests
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# Subdirectory Integration
# 1. Baseline: Header-only logic
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Graph Analytics Engine Base")


# 2. Optimized: Infrastructure and source implementations
add_subdirectory("Graph Analytics Engine Main")

# Enable Testing globally
enable_testing()
