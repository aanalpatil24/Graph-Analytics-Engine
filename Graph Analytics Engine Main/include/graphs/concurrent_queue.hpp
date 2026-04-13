// include/graph/concurrent_queue.hpp
#pragma once
#include <atomic>
#include <memory>
#include <optional>
#include <cstdint>

namespace graph::concurrent {

// Lock-free Michael-Scott Queue adapted for uint32_t node indices
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

    // Enqueue a value
    void push(uint32_t value);
    
    // Dequeue a value, returns nullopt if empty
    [[nodiscard]] std::optional<uint32_t> pop();
    
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

    // Disable copying/moving
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
};

// Work-stealing queue for thread pool
class WorkStealingQueue {
public:
    explicit WorkStealingQueue(size_t capacity = 1024);
    
    bool push(uint32_t item);
    std::optional<uint32_t> pop();      // Owner thread
    std::optional<uint32_t> steal();    // Other threads
    
    [[nodiscard]] bool empty() const noexcept;

private:
    struct alignas(64) Item {
        std::atomic<uint32_t> value;
        std::atomic<bool> valid{false};
    };
    
    std::vector<Item> buffer_;
    alignas(64) std::atomic<size_t> top_{0};
    alignas(64) std::atomic<size_t> bottom_{0};
    const size_t capacity_;
    const size_t mask_;
};

}