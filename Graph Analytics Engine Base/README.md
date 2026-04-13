# Graph Analytics Engine (Baseline Implementation)

## Description
This module serves as the standard, object-oriented baseline for the Safe Flow Decomposition project. It implements graph traversal, shortest-path (SPFA), and maximal safe path identification using traditional C++20 enterprise design patterns. It utilizes `std::vector` backed adjacency lists and standard `std::mutex` thread pooling. 

This implementation prioritizes generic abstractions and standard library conveniences over hardware sympathy, serving as the control group to benchmark against the hardware-accelerated Graph Analytics Engine.

## Outcomes & Benchmarks
* **Algorithmic Correctness:** Successfully implements the core mathematical logic for capacity-constrained flow decomposition.
* **Baseline Profiling:** Profiling this architecture via Callgrind revealed heavy L1/L2 cache misses due to pointer chasing in the adjacency list, and severe thread contention from atomic reference counting (`std::shared_ptr`) in the queue.
* **The Benchmark Target:** Establishes the baseline execution time that the AVX-512 engine must beat.

## Tech Stack
* **Language:** C++20 (Concepts, Ranges)
* **Data Structures:** Standard STL Adjacency Lists (`std::vector<std::vector<Edge>>`)
* **Concurrency:** `std::mutex`, `std::condition_variable`, `std::shared_ptr`
* **Validation:** Google Test (gTest)

## Quick Start

### Prerequisites
* CMake 3.25+
* GCC 12+ or Clang 14+ (C++20 support required)

### Build Instructions

``` bash
# 1. Create the build directory
mkdir build && cd build

# 2. Configure the project with strict warnings and sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SANITIZERS=ON

# 3. Compile the engine and test suite
cmake --build . -j $(nproc)

# 4. Run the validation suite
ctest --output-on-failure
```
