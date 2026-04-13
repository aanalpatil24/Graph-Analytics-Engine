GraphAnalyticsEngine/
├── base/                   <-- Standard O-O C++
│   ├── baseline_graph.hpp      # std::vector adjacency lists
│   ├── baseline_spfa.hpp       # Scalar loops
│   └── baseline_toposort.hpp   # Sequential Kahn's
├── main/                  # Hardware-Aware Engine
│   ├── include/
│   │   ├── graph/              # CSR Graph & Aligned Allocators
│   │   ├── simd/               # AVX-512 Intrinsics
│   │   └── utils/              # Lock-free Queues & Thread Pools
│   ├── src/                    # Optimized implementations
│   └── benchmarks/             # Code that races Baseline vs. Optimized
├── CMakeLists.txt              # Root build file connecting both folders
└── README.md                   # Highlights the 3-4x speedup