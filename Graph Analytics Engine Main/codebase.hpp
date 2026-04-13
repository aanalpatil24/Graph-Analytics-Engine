=== include/graph/aligned_allocator.hpp ===
#pragma once
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <limits>

namespace graph::memory {

/**
 * @brief A custom memory allocator that enforces strict byte alignment.
 * * AVX-512 registers (ZMM) are 512 bits (64 bytes) wide. If we attempt to load 
 * memory into these registers that crosses a 64-byte cache line boundary, the CPU 
 * suffers massive performance penalties or triggers a hardware segmentation fault.
 * This allocator ensures all graph data structures start perfectly aligned.
 */
template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    // Ensure std::vector can efficiently move allocations without copying data
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    // Enforce valid alignment parameters at compile-time
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");

    AlignedAllocator() noexcept = default;
    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        
        // std::aligned_alloc requires the requested size to be a multiple of the alignment
        void* ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        std::free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

} // namespace graph::memory

=== include/graph/csr_graph.hpp ===
#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <span>
#include "aligned_allocator.hpp"

namespace graph {

/**
 * @brief Compressed Sparse Row (CSR) Graph Representation.
 * * Standard adjacency lists (std::vector<std::vector<Edge>>) scatter memory across 
 * the heap, causing L1/L2 cache misses. CSR flattens the graph into contiguous 
 * arrays, maximizing cache locality and enabling SIMD hardware instructions.
 */
class CSRGraph {
public:
    // Typedefs utilizing our custom 64-byte aligned allocator
    using AlignedUint32Vec = std::vector<uint32_t, memory::AlignedAllocator<uint32_t, 64>>;
    using AlignedInt32Vec = std::vector<int32_t, memory::AlignedAllocator<int32_t, 64>>;
    using AlignedEdgeDestVec = std::vector<uint32_t, memory::AlignedAllocator<uint32_t, 64>>;

    CSRGraph() = default;
    
    // Strict move semantics: Graphs can be millions of edges. We disable copying
    // entirely to prevent accidental O(V+E) deep copies. Ownership is strictly moved.
    CSRGraph(CSRGraph&&) noexcept = default;
    CSRGraph& operator=(CSRGraph&&) noexcept = default;
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;

    CSRGraph(AlignedUint32Vec offsets, AlignedEdgeDestVec destinations, 
             AlignedInt32Vec weights, size_t num_vertices);

    [[nodiscard]] size_t num_vertices() const noexcept { return num_vertices_; }
    [[nodiscard]] size_t num_edges() const noexcept { return destinations_.size(); }
    
    // std::span provides safe, zero-cost boundary viewing into the contiguous arrays
    [[nodiscard]] std::span<const uint32_t> offsets() const noexcept {
        return std::span{offsets_.data(), offsets_.size()};
    }
    
    [[nodiscard]] std::span<const uint32_t> destinations() const noexcept {
        return std::span{destinations_.data(), destinations_.size()};
    }
    
    [[nodiscard]] std::span<const int32_t> weights() const noexcept {
        return std::span{weights_.data(), weights_.size()};
    }

    [[nodiscard]] std::span<const uint32_t> neighbors(uint32_t vertex) const noexcept {
        const auto start = offsets_[vertex];
        const auto end = offsets_[vertex + 1];
        return std::span{destinations_.data() + start, end - start};
    }

    [[nodiscard]] std::span<const int32_t> neighbor_weights(uint32_t vertex) const noexcept {
        const auto start = offsets_[vertex];
        const auto end = offsets_[vertex + 1];
        return std::span{weights_.data() + start, end - start};
    }

    // Direct raw pointer access required for passing to AVX-512 intrinsic functions
    [[nodiscard]] const uint32_t* aligned_offsets() const noexcept { return offsets_.data(); }
    [[nodiscard]] const uint32_t* aligned_destinations() const noexcept { return destinations_.data(); }
    [[nodiscard]] const int32_t* aligned_weights() const noexcept { return weights_.data(); }

private:
    AlignedUint32Vec offsets_;      // Starting index of neighbors for each vertex
    AlignedEdgeDestVec destinations_; // Flattened array of all neighbor IDs
    AlignedInt32Vec weights_;       // Flattened array of all edge weights
    size_t num_vertices_{0};
};

} // namespace graph

=== include/graph/graph_builder.hpp ===
#pragma once
#include <vector>
#include <utility>
#include "csr_graph.hpp"

namespace graph {

/**
 * @brief Mutable builder for constructing the immutable CSRGraph.
 * * Allows standard edge-by-edge insertion. Once construction is complete, 
 * calling build() flattens the data and transfers ownership to the CSRGraph.
 */
class GraphBuilder {
public:
    explicit GraphBuilder(size_t expected_vertices = 0, size_t expected_edges = 0);
    
    void add_edge(uint32_t from, uint32_t to, int32_t weight);
    void add_vertex();
    
    [[nodiscard]] size_t current_vertices() const noexcept { return adjacency_.size(); }
    [[nodiscard]] size_t current_edges() const noexcept { return total_edges_; }

    // Consumes the builder and returns the optimized, memory-aligned CSRGraph
    [[nodiscard]] CSRGraph build();
    void clear();

private:
    std::vector<std::vector<std::pair<uint32_t, int32_t>>> adjacency_;
    size_t total_edges_{0};
};

} // namespace graph

=== include/graph/concurrent_queue.hpp ===
#pragma once
#include <atomic>
#include <memory>
#include <optional>
#include <cstdint>

namespace graph::concurrent {

/**
 * @brief A Michael-Scott Lock-Free Queue.
 * * Eliminates std::mutex locking. Uses Atomic Compare-And-Swap (CAS) to allow 
 * multiple threads to enqueue and dequeue concurrently without thread stalls 
 * or kernel-level context switching.
 */
class LockFreeQueue {
private:
    struct Node {
        std::atomic<uint32_t> data;
        std::atomic<Node*> next;
        
        Node(uint32_t val) : data(val), next(nullptr) {}
    };

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    std::atomic<size_t> size_;

public:
    LockFreeQueue();
    ~LockFreeQueue();

    void push(uint32_t value);
    [[nodiscard]] std::optional<uint32_t> pop();
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

    // Prevent copying to maintain thread safety guarantees
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
};

/**
 * @brief Wait-free, fixed-capacity ring buffer for Work-Stealing.
 * * Used by individual threads to manage their local workload. If a thread 
 * runs out of work, it can "steal" from the top of another thread's queue.
 */
class WorkStealingQueue {
public:
    explicit WorkStealingQueue(size_t capacity = 1024);
    
    bool push(uint32_t item);
    std::optional<uint32_t> pop();     // Thread pops from its own bottom
    std::optional<uint32_t> steal();   // Other threads steal from the top
    [[nodiscard]] bool empty() const noexcept;

private:
    // Align to 64 bytes to prevent false-sharing across cache lines between threads
    struct alignas(64) Item {
        std::atomic<uint32_t> value;
        std::atomic<bool> valid{false};
    };
    
    std::vector<Item> buffer_;
    alignas(64) std::atomic<size_t> top_{0};
    alignas(64) std::atomic<size_t> bottom_{0};
    const size_t capacity_;
    const size_t mask_; // Used for fast modulo arithmetic (capacity must be power of 2)
};

} // namespace graph::concurrent

=== include/graph/thread_pool.hpp ===
#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace graph {

/**
 * @brief Standard worker pool for executing parallel graph tasks.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    [[nodiscard]] size_t size() const noexcept { return workers_.size(); }
    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
};

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}

/**
 * @brief Utility to split iterations evenly across the thread pool.
 */
class ParallelExecutor {
public:
    explicit ParallelExecutor(ThreadPool& pool) : pool_(pool) {}

    template<typename IndexFunc>
    void parallel_for(uint32_t start, uint32_t end, IndexFunc&& func);

    template<typename RangeFunc>
    void parallel_range(uint32_t start, uint32_t end, RangeFunc&& func);

private:
    ThreadPool& pool_;
};

template<typename IndexFunc>
void ParallelExecutor::parallel_for(uint32_t start, uint32_t end, IndexFunc&& func) {
    size_t n = end - start;
    size_t num_threads = pool_.size();
    size_t chunk = (n + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        uint32_t s = start + t * chunk;
        uint32_t e = std::min(start + (t + 1) * chunk, end);
        if (s < e) {
            futures.push_back(pool_.enqueue([func, s, e]() {
                for (uint32_t i = s; i < e; ++i) {
                    func(i);
                }
            }));
        }
    }
    
    for (auto& f : futures) f.wait();
}

} // namespace graph

=== include/graph/spfa_engine.hpp ===
#pragma once
#include <vector>
#include <atomic>
#include <limits>
#include <memory>
#include "csr_graph.hpp"
#include "concurrent_queue.hpp"
#include "thread_pool.hpp"

namespace graph {

/**
 * @brief Contains output data from a Shortest Path Faster Algorithm execution.
 * Uses atomic integers to allow concurrent discovery updates from multiple threads.
 */
struct SPFAResult {
    std::vector<std::atomic<int32_t>> distances;
    std::vector<std::atomic<uint32_t>> predecessors;
    bool success;
    
    explicit SPFAResult(size_t n) 
        : distances(n), predecessors(n), success(false) {
        for (auto& d : distances) d.store(std::numeric_limits<int32_t>::max());
        for (auto& p : predecessors) p.store(std::numeric_limits<uint32_t>::max());
    }
};

/**
 * @brief Core engine for shortest path calculations.
 * Features 3 modes: Scalar (baseline), Vectorized (AVX-512), and Concurrent (Multi-core).
 */
class SPFASolver {
public:
    explicit SPFASolver(const CSRGraph& graph, ThreadPool& pool);
    
    [[nodiscard]] SPFAResult solve_scalar(uint32_t source);
    [[nodiscard]] SPFAResult solve_vectorized(uint32_t source);
    [[nodiscard]] SPFAResult solve_concurrent(uint32_t source, size_t num_workers = 0);

private:
    const CSRGraph& graph_;
    ThreadPool& pool_;
    
    void relax_edges_avx512(uint32_t u, 
                           std::atomic<int32_t>* distances,
                           concurrent::LockFreeQueue& queue,
                           std::atomic<bool>* in_queue);
    
    void relax_edges_scalar(uint32_t u, uint32_t start, uint32_t end,
                           std::atomic<int32_t>* distances,
                           concurrent::LockFreeQueue& queue,
                           std::atomic<bool>* in_queue);
};

} // namespace graph

=== include/graph/cycle_detector.hpp ===
#pragma once
#include <vector>
#include <atomic>
#include <optional>
#include "csr_graph.hpp"
#include "thread_pool.hpp"

namespace graph {

struct CycleInfo {
    bool has_cycle;
    std::vector<uint32_t> cycle_nodes;
    std::vector<uint32_t> topological_order;
};

class CycleDetector {
public:
    explicit CycleDetector(const CSRGraph& graph, ThreadPool& pool);
    
    [[nodiscard]] CycleInfo detect_and_resolve();
    [[nodiscard]] std::vector<uint32_t> find_cycle_components(
        const std::vector<bool>& unvisited_mask);

private:
    const CSRGraph& graph_;
    ThreadPool& pool_;
    
    void parallel_dfs_collect(uint32_t start, 
                             std::vector<bool>& visited,
                             std::vector<uint32_t>& component);
};

} // namespace graph

=== src/csr_graph.cpp ===
#include "graph/csr_graph.hpp"

namespace graph {

CSRGraph::CSRGraph(AlignedUint32Vec offsets, AlignedEdgeDestVec destinations,
                   AlignedInt32Vec weights, size_t num_vertices)
    : offsets_(std::move(offsets))
    , destinations_(std::move(destinations))
    , weights_(std::move(weights))
    , num_vertices_(num_vertices) {
    
    // Safety check: Ensure the OS actually granted 64-byte aligned memory.
    // If not, AVX-512 intrinsic instructions will crash the program.
    if (reinterpret_cast<uintptr_t>(offsets_.data()) % 64 != 0 ||
        reinterpret_cast<uintptr_t>(destinations_.data()) % 64 != 0 ||
        reinterpret_cast<uintptr_t>(weights_.data()) % 64 != 0) {
        throw std::runtime_error("CSRGraph arrays not 64-byte aligned");
    }
}

} // namespace graph

=== src/graph_builder.cpp ===
#include "graph/graph_builder.hpp"
#include <algorithm>

namespace graph {

GraphBuilder::GraphBuilder(size_t expected_vertices, size_t expected_edges) {
    adjacency_.reserve(expected_vertices);
    for (size_t i = 0; i < expected_vertices; ++i) {
        adjacency_.emplace_back();
        adjacency_.back().reserve(expected_edges / std::max(expected_vertices, size_t(1)));
    }
}

void GraphBuilder::add_vertex() {
    adjacency_.emplace_back();
}

void GraphBuilder::add_edge(uint32_t from, uint32_t to, int32_t weight) {
    if (from >= adjacency_.size()) {
        adjacency_.resize(from + 1);
    }
    adjacency_[from].emplace_back(to, weight);
    ++total_edges_;
}

CSRGraph GraphBuilder::build() {
    const size_t n = adjacency_.size();
    const size_t m = total_edges_;
    
    CSRGraph::AlignedUint32Vec offsets(n + 1);
    CSRGraph::AlignedEdgeDestVec destinations;
    CSRGraph::AlignedInt32Vec weights;
    
    destinations.reserve(m);
    weights.reserve(m);
    
    offsets[0] = 0;
    
    // Construct the CSR format: Offsets track where a vertex's edges begin
    for (size_t i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + adjacency_[i].size();
    }
    
    // Flatten the adjacency list into the contiguous destinations and weights arrays
    for (auto& vec : adjacency_) {
        std::sort(vec.begin(), vec.end(), 
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        for (auto& [dest, weight] : vec) {
            destinations.push_back(dest);
            weights.push_back(weight);
        }
    }
    
    // Free up builder memory
    adjacency_.clear();
    total_edges_ = 0;
    
    // Return CSRGraph using std::move to prevent copying the massive contiguous arrays
    return CSRGraph(std::move(offsets), std::move(destinations), 
                    std::move(weights), n);
}

void GraphBuilder::clear() {
    adjacency_.clear();
    total_edges_ = 0;
}

} // namespace graph

=== src/concurrent_queue.cpp ===
#include "graph/concurrent_queue.hpp"
#include <stdexcept>

namespace graph::concurrent {

LockFreeQueue::LockFreeQueue() {
    Node* dummy = new Node(0);
    head_.store(dummy);
    tail_.store(dummy);
    size_.store(0);
}

LockFreeQueue::~LockFreeQueue() {
    while (Node* old = head_.load()) {
        head_.store(old->next.load());
        delete old;
    }
}

void LockFreeQueue::push(uint32_t value) {
    Node* new_node = new Node(value);
    Node* tail;
    
    while (true) {
        // memory_order_acquire ensures we see the most recent queue state
        tail = tail_.load(std::memory_order_acquire);
        Node* next = tail->next.load(std::memory_order_acquire);
        
        // Double check tail hasn't changed
        if (tail == tail_.load(std::memory_order_acquire)) {
            if (next == nullptr) {
                // Attempt to link new_node using Compare-And-Swap (CAS)
                // memory_order_release makes this new node visible to other threads instantly
                if (tail->next.compare_exchange_weak(next, new_node,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                    break; // Successfully linked
                }
            } else {
                // Another thread got here first but didn't update tail. Help them out.
                tail_.compare_exchange_weak(tail, next,
                                           std::memory_order_release,
                                           std::memory_order_relaxed);
            }
        }
    }
    
    // Update the tail pointer to the newly inserted node
    tail_.compare_exchange_strong(tail, new_node,
                                 std::memory_order_release,
                                 std::memory_order_relaxed);
    size_.fetch_add(1, std::memory_order_relaxed);
}

std::optional<uint32_t> LockFreeQueue::pop() {
    Node* head;
    
    while (true) {
        head = head_.load(std::memory_order_acquire);
        Node* tail = tail_.load(std::memory_order_acquire);
        Node* next = head->next.load(std::memory_order_acquire);
        
        if (head == head_.load(std::memory_order_acquire)) {
            if (head == tail) {
                if (next == nullptr) {
                    return std::nullopt; // Queue is genuinely empty
                }
                // Tail is falling behind, help update it
                tail_.compare_exchange_weak(tail, next,
                                           std::memory_order_release,
                                           std::memory_order_relaxed);
            } else {
                // Read value before unlinking node
                uint32_t value = next->data.load(std::memory_order_relaxed);
                // Attempt to swing head to the next node
                if (head_.compare_exchange_weak(head, next,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    delete head; // Reclaim memory
                    return value;
                }
            }
        }
    }
}

bool LockFreeQueue::empty() const noexcept {
    Node* head = head_.load(std::memory_order_acquire);
    Node* next = head->next.load(std::memory_order_acquire);
    return next == nullptr;
}

size_t LockFreeQueue::size() const noexcept {
    return size_.load(std::memory_order_relaxed);
}

WorkStealingQueue::WorkStealingQueue(size_t capacity) 
    : buffer_(capacity), capacity_(capacity), mask_(capacity - 1) {
    if ((capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument("Capacity must be power of 2");
    }
}

bool WorkStealingQueue::push(uint32_t item) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);
    
    if (b - t >= capacity_ - 1) {
        return false; // Queue full
    }
    
    buffer_[b & mask_].value.store(item, std::memory_order_relaxed);
    buffer_[b & mask_].valid.store(true, std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_release);
    return true;
}

std::optional<uint32_t> WorkStealingQueue::pop() {
    size_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    
    // Sequential consistency fence to synchronize with stealing threads
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t t = top_.load(std::memory_order_relaxed);
    
    if (t <= b) {
        uint32_t item = buffer_[b & mask_].value.load(std::memory_order_relaxed);
        if (t == b) {
            // Last item in queue, race against a potential steal
            if (!top_.compare_exchange_strong(t, t + 1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                bottom_.store(b + 1, std::memory_order_relaxed);
                return std::nullopt; // Stealer won the race
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return item;
    } else {
        bottom_.store(b + 1, std::memory_order_relaxed); // Queue was empty
        return std::nullopt;
    }
}

std::optional<uint32_t> WorkStealingQueue::steal() {
    size_t t = top_.load(std::memory_order_acquire);
    // Fence required to prevent instruction reordering around the top/bottom load
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);
    
    if (t < b) {
        uint32_t item = buffer_[t & mask_].value.load(std::memory_order_relaxed);
        // Attempt to steal the top item via CAS
        if (top_.compare_exchange_strong(t, t + 1,
                                        std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
            return item;
        }
    }
    return std::nullopt;
}

bool WorkStealingQueue::empty() const noexcept {
    return top_.load(std::memory_order_acquire) >= 
           bottom_.load(std::memory_order_acquire);
}

} // namespace graph::concurrent

=== src/thread_pool.cpp ===
#include "graph/thread_pool.hpp"

namespace graph {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) return;
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task(); // Execute fetched task
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_.store(true);
    }
    condition_.notify_all();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace graph

=== src/spfa_engine.cpp ===
#include "graph/spfa_engine.hpp"
#include <immintrin.h> // Intel AVX-512 Intrinsics
#include <cstring>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>

namespace graph {

SPFASolver::SPFASolver(const CSRGraph& graph, ThreadPool& pool) 
    : graph_(graph), pool_(pool) {}

SPFAResult SPFASolver::solve_scalar(uint32_t source) {
    // Standard scalar implementation of Shortest Path Faster Algorithm (SPFA).
    // Serves as the correctness and performance baseline.
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
            
            // Relax edge
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
    SPFAResult result(graph_.num_vertices());
    const size_t n = graph_.num_vertices();
    
    result.distances[source].store(0);
    
    std::vector<uint32_t> queue;
    queue.reserve(n);
    std::vector<bool> in_queue(n, false);
    
    queue.push_back(source);
    in_queue[source] = true;
    size_t qhead = 0;
    
    // Obtain 64-byte aligned pointers for safe AVX-512 memory loads
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
    
    const auto* offsets = graph_.aligned_offsets();
    const auto* dests = graph_.aligned_destinations();
    const auto* weights = graph_.aligned_weights();
    
    auto worker_func = [&]() {
        active_workers.fetch_add(1);
        
        // memory_order_acquire pairs with memory_order_release below to ensure state synchronization
        while (!done.load(std::memory_order_acquire)) {
            auto opt_v = queue.pop();
            
            if (opt_v) {
                uint32_t u = *opt_v;
                in_queue[u].store(false, std::memory_order_release);
                
                int32_t du = result.distances[u].load(std::memory_order_acquire);
                if (du == std::numeric_limits<int32_t>::max()) continue;
                
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
    
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, worker_func));
    }
    
    for (auto& f : futures) f.wait();
    
    result.success = true;
    return result;
}

} // namespace graph

=== src/cycle_detector.cpp ===
#include "graph/cycle_detector.hpp"
#include <stack>
#include <algorithm>

namespace graph {

CycleDetector::CycleDetector(const CSRGraph& graph, ThreadPool& pool) 
    : graph_(graph), pool_(pool) {}

CycleInfo CycleDetector::detect_and_resolve() {
    CycleInfo info;
    const size_t n = graph_.num_vertices();
    
    // Track in-degrees atomically to support parallel computation
    std::vector<std::atomic<uint32_t>> in_degree(n);
    for (size_t i = 0; i < n; ++i) in_degree[i].store(0);
    
    // Parallel computation of initial in-degrees
    ParallelExecutor executor(pool_);
    executor.parallel_for(0, n, [&](uint32_t u) {
        auto neighbors = graph_.neighbors(u);
        for (uint32_t v : neighbors) {
            in_degree[v].fetch_add(1, std::memory_order_relaxed);
        }
    });
    
    concurrent::LockFreeQueue zero_queue;
    std::vector<bool> processed(n, false);
    
    // Push all starting nodes (in-degree 0) into lock-free queue
    for (uint32_t i = 0; i < n; ++i) {
        if (in_degree[i].load(std::memory_order_relaxed) == 0) {
            zero_queue.push(i);
        }
    }
    
    std::mutex topo_mutex;
    std::atomic<size_t> processed_count{0};
    
    // Parallel implementation of Kahn's Topological Sort
    auto worker = [&]() {
        while (true) {
            auto opt_u = zero_queue.pop();
            if (!opt_u) {
                if (processed_count.load(std::memory_order_acquire) >= n) break;
                continue;
            }
            
            uint32_t u = *opt_u;
            if (processed[u]) continue;
            processed[u] = true;
            
            {
                std::lock_guard<std::mutex> lock(topo_mutex);
                info.topological_order.push_back(u);
            }
            processed_count.fetch_add(1);
            
            auto neighbors = graph_.neighbors(u);
            for (uint32_t v : neighbors) {
                // Atomically decrement dependency count. If 0, node is ready.
                uint32_t new_deg = in_degree[v].fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (new_deg == 0) {
                    zero_queue.push(v);
                }
            }
        }
    };
    
    size_t num_workers = std::min(pool_.size(), size_t(4));
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_workers; ++i) {
        futures.push_back(std::async(std::launch::async, worker));
    }
    for (auto& f : futures) f.wait();
    
    // If we didn't process all nodes, a cyclic dependency exists
    info.has_cycle = (info.topological_order.size() != n);
    
    if (info.has_cycle) {
        std::vector<bool> unvisited(n, false);
        for (uint32_t i = 0; i < n; ++i) {
            if (!processed[i]) unvisited[i] = true;
        }
        
        info.cycle_nodes = find_cycle_components(unvisited);
    }
    
    return info;
}

std::vector<uint32_t> CycleDetector::find_cycle_components(
    const std::vector<bool>& unvisited_mask) {
    
    std::vector<uint32_t> cycles;
    const size_t n = graph_.num_vertices();
    std::vector<bool> visited(n, false);
    
    ParallelExecutor executor(pool_);
    
    // Isolate nodes participating in the cycle
    for (uint32_t start = 0; start < n; ++start) {
        if (!unvisited_mask[start] || visited[start]) continue;
        
        std::vector<uint32_t> local_stack;
        std::vector<uint32_t> local_component;
        local_stack.push_back(start);
        
        while (!local_stack.empty()) {
            uint32_t u = local_stack.back();
            local_stack.pop_back();
            
            if (visited[u]) continue;
            visited[u] = true;
            local_component.push_back(u);
            
            auto neighbors = graph_.neighbors(u);
            for (uint32_t v : neighbors) {
                if (unvisited_mask[v] && !visited[v]) {
                    local_stack.push_back(v);
                }
            }
        }
        
        if (!local_component.empty()) {
            cycles.insert(cycles.end(), local_component.begin(), local_component.end());
        }
    }
    
    return cycles;
}

} // namespace graph

=== tests/test_graph.cpp ===
#include <gtest/gtest.h>
#include "graph/csr_graph.hpp"
#include "graph/graph_builder.hpp"
#include "graph/spfa_engine.hpp"
#include "graph/cycle_detector.hpp"
#include "graph/thread_pool.hpp"
#include "graph/concurrent_queue.hpp"
#include <limits>
#include <thread>
#include <atomic>

using namespace graph;

// Verifies that the custom allocator is successfully enforcing the 64-byte 
// boundary required to prevent AVX-512 hardware faults.
TEST(CSRGraphTest, MemoryAlignment) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 5);
    builder.add_edge(0, 2, 3);
    
    auto graph = builder.build();
    
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.aligned_offsets()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.aligned_destinations()) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(graph.aligned_weights()) % 64, 0);
}

// Verifies that both the AVX-512 implementation and the concurrent thread-pool 
// implementation yield mathematically identical results to the scalar baseline.
TEST(SPFATest, Correctness) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 6);
    builder.add_edge(0, 2, 7);
    builder.add_edge(1, 2, 8);
    builder.add_edge(1, 3, 5);
    builder.add_edge(1, 4, -4);
    builder.add_edge(2, 3, -3);
    builder.add_edge(2, 4, 9);
    builder.add_edge(3, 1, -2);
    builder.add_edge(4, 0, 2);
    builder.add_edge(4, 3, 7);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    SPFASolver solver(graph, pool);
    
    auto result_scalar = solver.solve_scalar(0);
    auto result_vec = solver.solve_vectorized(0);
    auto result_conc = solver.solve_concurrent(0, 4);
    
    EXPECT_EQ(result_scalar.distances[4].load(), -2);
    
    for (size_t i = 0; i < graph.num_vertices(); ++i) {
        EXPECT_EQ(result_vec.distances[i].load(), result_scalar.distances[i].load());
        EXPECT_EQ(result_conc.distances[i].load(), result_scalar.distances[i].load());
    }
}

// Stress-tests the Michael-Scott compare-and-swap logic to ensure no race 
// conditions or lost data occurs under heavy concurrent pressure.
TEST(ConcurrentQueueTest, LockFree) {
    concurrent::LockFreeQueue queue;
    std::atomic<size_t> pushed{0};
    std::atomic<size_t> popped{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1000; ++i) {
                queue.push(i);
                pushed.fetch_add(1);
            }
        });
    }
    
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            int local = 0;
            while (local < 1000) {
                auto val = queue.pop();
                if (val) {
                    popped.fetch_add(1);
                    ++local;
                }
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    EXPECT_EQ(pushed.load(), 4000);
    EXPECT_EQ(popped.load(), 4000);
}

TEST(CycleDetectionTest, DAG) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    builder.add_edge(2, 3, 1);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    CycleDetector detector(graph, pool);
    
    auto info = detector.detect_and_resolve();
    
    EXPECT_FALSE(info.has_cycle);
    EXPECT_EQ(info.topological_order.size(), 4);
}

TEST(CycleDetectionTest, WithCycle) {
    GraphBuilder builder;
    builder.add_edge(0, 1, 1);
    builder.add_edge(1, 2, 1);
    builder.add_edge(2, 0, 1);
    builder.add_edge(0, 3, 1);
    
    auto graph = builder.build();
    ThreadPool pool(4);
    CycleDetector detector(graph, pool);
    
    auto info = detector.detect_and_resolve();
    
    EXPECT_TRUE(info.has_cycle);
    EXPECT_EQ(info.cycle_nodes.size(), 3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

=== CMakeLists.txt ===
cmake_minimum_required(VERSION 3.20)
project(GraphAnalyticsEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -mavx512f and -mavx512vl explicitly enable AVX-512 support during compilation
    add_compile_options(-O3 -mavx512f -mavx512vl -march=native -mtune=native)
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX512 /W4)
endif()

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

add_library(graph_engine STATIC
    src/csr_graph.cpp
    src/graph_builder.cpp
    src/concurrent_queue.cpp
    src/thread_pool.cpp
    src/spfa_engine.cpp
    src/cycle_detector.cpp
)

target_include_directories(graph_engine PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(graph_engine PUBLIC Threads::Threads)

enable_testing()
add_executable(graph_tests tests/test_graph.cpp)
target_link_libraries(graph_tests PRIVATE graph_engine GTest::gtest)
gtest_discover_tests(graph_tests)