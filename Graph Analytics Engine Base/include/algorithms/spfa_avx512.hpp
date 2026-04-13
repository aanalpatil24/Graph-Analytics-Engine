// include/algorithms/spfa_avx512.hpp
#pragma once
#include <immintrin.h>
#include <vector>
#include <atomic>
#include <limits>
#include <cstdint>
#include "graph/csr_graph.hpp"
#include "utils/lock_free_queue.hpp"
#include "utils/thread_pool.hpp"

namespace graph {

/**
 * AVX-512 optimized SPFA (Shortest Path Faster Algorithm)
 * Vectorized relaxation processing 16 edges simultaneously
 */
class AVX512SPFA {
private:
    const CSRGraph& graph_;
    std::vector<std::atomic<Weight>> dist_;
    std::vector<std::atomic<bool>> in_queue_;
    LockFreeQueue<VertexId> queue_;
    
    // Check if AVX-512 is available at runtime
    static bool has_avx512() {
        // Simplified check - in production use CPUID
        return __builtin_cpu_supports("avx512f");
    }

public:
    explicit AVX512SPFA(const CSRGraph& graph) 
        : graph_(graph)
        , dist_(graph.num_vertices())
        , in_queue_(graph.num_vertices()) {}

    std::vector<Weight> compute_shortest_paths(VertexId source) {
        const size_t n = graph_.num_vertices();
        
        // Initialize distances
        for (auto& d : dist_) {
            d.store(INF_WEIGHT, std::memory_order_relaxed);
        }
        dist_[source].store(0, std::memory_order_relaxed);
        
        // Initialize queue with source
        for (auto& flag : in_queue_) {
            flag.store(false, std::memory_order_relaxed);
        }
        
        queue_.enqueue(source);
        in_queue_[source].store(true, std::memory_order_relaxed);
        
        if (has_avx512() && n >= 16) {
            run_vectorized();
        } else {
            run_scalar();
        }
        
        // Extract results
        std::vector<Weight> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist_[i].load(std::memory_order_acquire);
        }
        return result;
    }

private:
    void run_scalar() {
        while (auto opt_v = queue_.dequeue()) {
            VertexId u = *opt_v;
            in_queue_[u].store(false, std::memory_order_release);
            
            auto [dsts, weights] = graph_.edges(u);
            Weight du = dist_[u].load(std::memory_order_acquire);
            
            for (size_t i = 0; i < dsts.size(); ++i) {
                VertexId v = dsts[i];
                Weight w = weights[i];
                Weight new_dist = du + w;
                
                Weight dv = dist_[v].load(std::memory_order_relaxed);
                while (new_dist < dv) {
                    if (dist_[v].compare_exchange_weak(
                            dv, new_dist,
                            std::memory_order_release,
                            std::memory_order_relaxed)) {
                        if (!in_queue_[v].exchange(true, std::memory_order_acquire)) {
                            queue_.enqueue(v);
                        }
                        break;
                    }
                }
            }
        }
    }

    void run_vectorized() {
        while (auto opt_v = queue_.dequeue()) {
            VertexId u = *opt_v;
            in_queue_[u].store(false, std::memory_order_release);
            
            auto [dsts, weights] = graph_.edges(u);
            const size_t degree = dsts.size();
            Weight du = dist_[u].load(std::memory_order_acquire);
            
            size_t i = 0;
            // Process 16 edges at a time using AVX-512
            for (; i + 16 <= degree; i += 16) {
                process_chunk_avx512(&dsts[i], &weights[i], du);
            }
            
            // Handle remainder with scalar code
            for (; i < degree; ++i) {
                relax_edge(dsts[i], weights[i], du);
            }
        }
    }

    void process_chunk_avx512(const VertexId* dsts, const Weight* weights, Weight du) {
        // Load 16 destination indices
        __m512i v_indices = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(dsts));
        
        // Load 16 edge weights
        __m512i v_weights = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights));
        
        // Broadcast source distance to all lanes
        __m512i v_du = _mm512_set1_epi32(du);
        
        // Compute new_distances = du + weight
        __m512i v_new_dist = _mm512_add_epi32(v_du, v_weights);
        
        // Gather current distances for destinations
        __m512i v_old_dist = _mm512_i32gather_epi32(
            v_indices, dist_.data(), sizeof(Weight)
        );
        
        // Compare: new_dist < old_dist
        __mmask16 mask = _mm512_cmp_epi32_mask(
            v_new_dist, v_old_dist, _MM_CMPINT_LT
        );
        
        if (mask == 0) return; // No updates needed
        
        // Scatter updated distances where mask is set
        _mm512_mask_i32scatter_epi32(
            dist_.data(), mask, v_indices, v_new_dist, sizeof(Weight)
        );
        
        // For each updated vertex, enqueue if not already in queue
        // Note: In practice, we'd want to batch enqueue operations
        // This scalar loop handles the mask extraction
        uint16_t m = mask;
        while (m) {
            int bit = __builtin_ctz(m);
            VertexId v = dsts[bit];
            if (!in_queue_[v].exchange(true, std::memory_order_acquire)) {
                queue_.enqueue(v);
            }
            m &= m - 1; // Clear lowest set bit
        }
    }

    void relax_edge(VertexId v, Weight w, Weight du) {
        Weight new_dist = du + w;
        Weight dv = dist_[v].load(std::memory_order_relaxed);
        
        while (new_dist < dv) {
            if (dist_[v].compare_exchange_weak(
                    dv, new_dist,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                if (!in_queue_[v].exchange(true, std::memory_order_acquire)) {
                    queue_.enqueue(v);
                }
                break;
            }
        }
    }
};

} // namespace graph