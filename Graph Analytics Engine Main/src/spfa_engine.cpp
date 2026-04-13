// src/spfa_engine.cpp
#include "graphs/spfa_engine.hpp"
#include "simd/avx512_utils.hpp"
#include <cstring>
#include <vector>
#include <queue>
#include <limits>

#ifdef ENABLE_PROFILING
#include <valgrind/callgrind.h>
#endif

namespace graph {

SPFASolver::SPFASolver(const CSRGraph& graph, ThreadPool& pool) 
    : graph_(graph), pool_(pool) {}

SPFAResult SPFASolver::solve_scalar(uint32_t source) {
    SPFAResult result(graph_.num_vertices());
    const size_t n = graph_.num_vertices();
    
    result.distances[source].store(0);
    
    std::queue<uint32_t> q;
    std::vector<bool> in_queue(n, false);
    
    q.push(source);
    in_queue[source] = true;
    
    while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();
        in_queue[u] = false;
        
        int32_t du = result.distances[u].load(std::memory_order_relaxed);
        
        auto neighbors = graph_.neighbors(u);
        auto weights = graph_.neighbor_weights(u);
        
        for (size_t i = 0; i < neighbors.size(); ++i) {
            uint32_t v = neighbors[i];
            int32_t w = weights[i];
            int32_t dv = result.distances[v].load(std::memory_order_relaxed);
            
            if (du != std::numeric_limits<int32_t>::max() && du + w < dv) {
                result.distances[v].store(du + w, std::memory_order_relaxed);
                result.predecessors[v].store(u, std::memory_order_relaxed);
                
                if (!in_queue[v]) {
                    q.push(v);
                    in_queue[v] = true;
                }
            }
        }
    }
    
    result.success = true;
    return result;
}

SPFAResult SPFASolver::solve_vectorized(uint32_t source) {
#ifdef ENABLE_PROFILING
    CALLGRIND_START_INSTRUMENTATION;
#endif

    SPFAResult result(graph_.num_vertices());
    const size_t n = graph_.num_vertices();
    
    result.distances[source].store(0);
    
    std::vector<uint32_t> queue;
    queue.reserve(n);
    std::vector<bool> in_queue(n, false);
    
    queue.push_back(source);
    in_queue[source] = true;
    size_t qhead = 0;
    
    const auto* offsets = graph_.aligned_offsets();
    const auto* dests = graph_.aligned_destinations();
    const auto* weights = graph_.aligned_weights();
    
    while (qhead < queue.size()) {
        uint32_t u = queue[qhead++];
        in_queue[u] = false;
        
        int32_t du = result.distances[u].load(std::memory_order_relaxed);
        
        uint32_t start = offsets[u];
        uint32_t end = offsets[u + 1];
        uint32_t count = end - start;
        
        uint32_t i = 0;
        
        // --- TRUE AVX-512 HARDWARE GATHER/SCATTER ---
        // Process up to 16 edges simultaneously in a single CPU clock cycle.
        if (simd::avx512_supported() && count >= 16) {
            for (; i + 16 <= count; i += 16) {
                uint32_t base = start + i;
                
                // 1. LOAD: Fetch 16 neighbor destinations and their edge weights into 512-bit ZMM registers.
                __m512i vdests = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(dests + base));
                __m512i vweights = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights + base));
                
                // 2. BROADCAST: Copy the current source distance (du) into all 16 lanes.
                __m512i vdu = _mm512_set1_epi32(du);
                
                // 3. COMPUTE: Vectorized addition calculates new potential distances for all 16 neighbors.
                __m512i vnew = _mm512_add_epi32(vdu, vweights);
                
                // 4. GATHER: The memory controller parallelizes fetching the current shortest distances
                // from the non-contiguous neighbor addresses. (Scale factor '4' is sizeof(int32_t)).
                __m512i vcur = _mm512_i32gather_epi32(vdests, result.distances.data(), 4);
                
                // 5. COMPARE: Generates a 16-bit mask where a '1' indicates the new path is shorter.
                __mmask16 mask = _mm512_cmplt_epi32_mask(vnew, vcur);
                
                if (mask != 0) {
                    // 6. SCATTER: Hardware safely writes only the updated distances back to memory.
                    _mm512_mask_i32scatter_epi32(result.distances.data(), mask, vdests, vnew, 4);
                    
                    // 7. O(K) EXTRACTION: Instead of looping 16 times to check the mask,
                    // Count Trailing Zeros (__builtin_ctz) jumps instantly to the successfully updated lanes.
                    uint16_t m = mask;
                    while (m) {
                        int bit = __builtin_ctz(m);
                        uint32_t v = dests[base + bit];
                        result.predecessors[v].store(u, std::memory_order_relaxed);
                        
                        if (!in_queue[v]) {
                            queue.push_back(v);
                            in_queue[v] = true;
                        }
                        m &= m - 1; // Clear the bit we just processed
                    }
                }
            }
        }
        
        // Scalar cleanup for vertices with < 16 remaining edges
        for (; i < count; ++i) {
            uint32_t v = dests[start + i];
            int32_t w = weights[start + i];
            int32_t dv = result.distances[v].load(std::memory_order_relaxed);
            
            if (du != std::numeric_limits<int32_t>::max() && du + w < dv) {
                result.distances[v].store(du + w, std::memory_order_relaxed);
                result.predecessors[v].store(u, std::memory_order_relaxed);
                
                if (!in_queue[v]) {
                    queue.push_back(v);
                    in_queue[v] = true;
                }
            }
        }
    }
    
    result.success = true;
    
#ifdef ENABLE_PROFILING
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    
    return result;
}

SPFAResult SPFASolver::solve_concurrent(uint32_t source, size_t num_workers) {
    if (num_workers == 0) num_workers = pool_.size();
    
    SPFAResult result(graph_.num_vertices());
    const size_t n = graph_.num_vertices();
    
    result.distances[source].store(0);
    
    // Utilize the Lock-Free Queue to prevent thread contention during edge discovery
    concurrent::LockFreeQueue queue;
    std::vector<std::atomic<bool>> in_queue(n);
    
    for (auto& flag : in_queue) flag.store(false, std::memory_order_relaxed);
    
    queue.push(source);
    in_queue[source].store(true, std::memory_order_relaxed);
    
    std::atomic<size_t> active_workers{0};
    std::atomic<bool> done{false};
    
    auto worker_func = [&](size_t worker_id) {
        active_workers.fetch_add(1);
        
        // memory_order_acquire pairs with memory_order_release below to ensure state synchronization
        while (!done.load(std::memory_order_acquire)) {
            auto opt_v = queue.pop();
            
            if (opt_v) {
                uint32_t u = *opt_v;
                in_queue[u].store(false, std::memory_order_release);
                
                int32_t du = result.distances[u].load(std::memory_order_acquire);
                if (du == std::numeric_limits<int32_t>::max()) continue;
                
                const auto* offsets = graph_.aligned_offsets();
                const auto* dests = graph_.aligned_destinations();
                const auto* weights = graph_.aligned_weights();
                
                uint32_t start = offsets[u];
                uint32_t end = offsets[u + 1];
                
                for (uint32_t i = start; i < end; ++i) {
                    uint32_t v = dests[i];
                    int32_t w = weights[i];
                    
                    int32_t current_dist = result.distances[v].load(std::memory_order_relaxed);
                    int32_t new_dist = du + w;
                    
                    if (new_dist < current_dist) {
                        // Atomic Compare-and-Swap resolves thread races on the same node update
                        while (!result.distances[v].compare_exchange_weak(
                                current_dist, new_dist,
                                std::memory_order_release,
                                std::memory_order_relaxed)) {
                            // If another thread wrote a smaller distance, bail out
                            if (current_dist <= new_dist) break;
                        }
                        
                        if (new_dist < current_dist) {
                            result.predecessors[v].store(u, std::memory_order_relaxed);
                            
                            // Prevent multiple threads from enqueuing the same node simultaneously
                            if (!in_queue[v].exchange(true, std::memory_order_acq_rel)) {
                                queue.push(v);
                            }
                        }
                    }
                }
            } else {
                // Cooperative termination detection: if all workers are idle, we are done
                if (active_workers.load(std::memory_order_acquire) == 1) {
                    done.store(true, std::memory_order_release);
                    break;
                }
                
                active_workers.fetch_sub(1);
                
                // Spin-wait optimization (prevents heavy kernel sleeps)
                for (int spin = 0; spin < 1000; ++spin) {
                    if (!queue.empty()) break;
                    _mm_pause(); // Intel pause instruction yields pipeline resources
                }
                
                if (queue.empty() && done.load(std::memory_order_acquire)) {
                    break;
                }
                
                active_workers.fetch_add(1);
            }
        }
        
        active_workers.fetch_sub(1);
    };
    
    // Launch workers
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, worker_func, i));
    }
    
    for (auto& f : futures) f.wait();
    
    result.success = true;
    return result;
}

}