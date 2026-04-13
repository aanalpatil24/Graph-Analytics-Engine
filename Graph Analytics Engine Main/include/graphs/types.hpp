// include/graph/types.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <limits>
#include <cstdlib>

namespace graph {

// Core Graph Types
using VertexId = int32_t;
using Weight = int32_t;
using EdgeId = int64_t;

constexpr Weight INF_WEIGHT = std::numeric_limits<Weight>::max() / 2;
constexpr VertexId NULL_VERTEX = -1;

// Cache line size for x86_64 architecture (Crucial for AVX-512 alignment)
constexpr size_t CACHE_LINE_SIZE = 64;

// Aligned allocation helper to prevent cache-line splitting
template<typename T>
T* aligned_alloc(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, count * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free(T* ptr) {
    free(ptr);
}

// Custom deleter for smart pointers ensuring aligned memory is freed correctly
template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        aligned_free(ptr);
    }
};

template<typename T>
using AlignedUniquePtr = std::unique_ptr<T[], AlignedDeleter<T>>;

} // namespace graph