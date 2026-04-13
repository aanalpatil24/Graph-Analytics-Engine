// src/simd/avx512_utils.cpp
#include "simd/avx512_utils.hpp"
#include <cpuid.h>

namespace graph::simd {

bool avx512_supported() {
    unsigned int eax, ebx, ecx, edx;
    
    // Check for OSXSAVE (bit 27 of ECX)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (!(ecx & bit_OSXSAVE)) return false;
    }
    
    // Check XCR0 for ZMM register state
    unsigned long long xcr0;
    asm("xgetbv" : "=A" (xcr0) : "c" (0) : "%edx");
    if ((xcr0 & 0xE6) != 0xE6) return false; // AVX-512 requires bits 5,6,7 and 1,2
    
    // Check AVX-512F (Foundation)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX512F) != 0;
    }
    
    return false;
}

void avx512_relax_edges(
    const int32_t* weights,
    const uint32_t* destinations,
    const int32_t* distances,
    int32_t current_dist,
    uint32_t base_idx,
    int32_t* out_distances,
    uint32_t* out_updates,
    size_t& out_count,
    size_t count) {
    
    if (count == 0) return;
    
    // Load weights and destinations
    __m512i vweights = _mm512_loadu_si512(weights);
    __m512i vdests = _mm512_loadu_si512(destinations);
    
    // Broadcast current distance
    __m512i vcur_dist = _mm512_set1_epi32(current_dist);
    
    // new_dist = current_dist + weight
    __m512i vnew_dist = _mm512_add_epi32(vcur_dist, vweights);
    
    // Gather current best distances (indices in vdests)
    __m512i vbest = _mm512_i32gather_epi32(vdests, distances, 4);
    
    // mask = new_dist < best_dist
    __mmask16 mask = _mm512_cmplt_epi32_mask(vnew_dist, vbest);
    
    if (mask != 0) {
        // Store updates
        _mm512_mask_compressstoreu_epi32(out_updates, mask, vdests);
        _mm512_mask_compressstoreu_epi32(out_distances, mask, vnew_dist);
        out_count = _mm_popcnt_u32(mask);
    } else {
        out_count = 0;
    }
}

}