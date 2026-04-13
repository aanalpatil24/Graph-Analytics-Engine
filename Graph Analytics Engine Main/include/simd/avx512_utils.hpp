// include/simd/avx512_utils.hpp
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

namespace graph::simd {

// Check if AVX-512 is available at runtime
bool avx512_supported();

// Vectorized relaxation using AVX-512
// Loads 16 edge weights and destinations, computes new distances, updates if smaller
void avx512_relax_edges(
    const int32_t* weights,           // Aligned to 64 bytes
    const uint32_t* destinations,     // Aligned to 64 bytes  
    const int32_t* distances,         // Source array for gather
    int32_t current_dist,
    uint32_t base_idx,
    int32_t* out_distances,           // For scatter updates
    uint32_t* out_updates,            // Buffer for nodes to update
    size_t& out_count,
    size_t count                      // Actual count (<=16)
);

// Gather-scatter based comparison and update
void avx512_update_distances(
    const uint32_t* destinations,
    const int32_t* weights,
    int32_t current_dist,
    int32_t* distance_array,
    uint8_t* update_mask              // Output: which indices to update
);

}