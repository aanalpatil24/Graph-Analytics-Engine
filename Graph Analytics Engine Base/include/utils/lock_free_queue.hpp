// include/utils/lock_free_queue.hpp
#pragma once
#include <atomic>
#include <memory>
#include <optional>
#include <new>
#include "graph/types.hpp"

namespace graph {

/**
 * Lock-free multi-producer multi-consumer queue based on Michael-Scott algorithm
 * Optimized for graph traversal work queues
 */
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };

    // Padding to prevent false sharing between head and tail
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> size_;

public:
    LockFreeQueue() {
        Node* dummy = new Node();
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
        size_.store(0, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        while (Node* old_head = head_.load(std::memory_order_relaxed)) {
            head_.store(old_head->next.load(std::memory_order_relaxed), 
                       std::memory_order_relaxed);
            delete old_head;
        }
    }

    // Delete copy/move to maintain thread safety invariants
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    void enqueue(T value) {
        T* p = new T(std::move(value));
        Node* new_node = new Node();
        new_node->data.store(p, std::memory_order_relaxed);
        
        Node* tail = tail_.load(std::memory_order_acquire);
        Node* next = tail->next.load(std::memory_order_acquire);
        
        while (true) {
            if (!next) {
                if (tail->next.compare_exchange_weak(
                        next, new_node,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    tail_.compare_exchange_strong(
                        tail, new_node,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                    size_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            } else {
                tail_.compare_exchange_weak(
                    tail, next,
                    std::memory_order_release,
                    std::memory_order_relaxed);
                tail = tail_.load(std::memory_order_acquire);
                next = tail->next.load(std::memory_order_acquire);
            }
        }
    }

    std::optional<T> dequeue() {
        Node* head = head_.load(std::memory_order_acquire);
        Node* tail = tail_.load(std::memory_order_acquire);
        Node* next = head->next.load(std::memory_order_acquire);
        
        while (true) {
            if (head == tail) {
                if (!next) return std::nullopt;
                tail_.compare_exchange_weak(
                    tail, next,
                    std::memory_order_release,
                    std::memory_order_relaxed);
            } else {
                if (head_.compare_exchange_weak(
                        head, next,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    T* p = next->data.load(std::memory_order_acquire);
                    T value = std::move(*p);
                    delete p;
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    delete head; // Safe deletion after CAS success
                    return value;
                }
                head = head_.load(std::memory_order_acquire);
                tail = tail_.load(std::memory_order_acquire);
                next = head->next.load(std::memory_order_acquire);
            }
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        Node* head = head_.load(std::memory_order_acquire);
        Node* tail = tail_.load(std::memory_order_acquire);
        return head == tail;
    }

    [[nodiscard]] size_t size() const noexcept {
        return size_.load(std::memory_order_relaxed);
    }
};

} // namespace graph