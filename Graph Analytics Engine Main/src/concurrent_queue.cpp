// src/concurrent_queue.cpp
#include "graph/concurrent_queue.hpp"
#include <stdexcept>

namespace graph::concurrent {

// LockFreeQueue Implementation (Michael-Scott)
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
        tail = tail_.load(std::memory_order_acquire);
        Node* next = tail->next.load(std::memory_order_acquire);
        
        if (tail == tail_.load(std::memory_order_acquire)) {
            if (next == nullptr) {
                if (tail->next.compare_exchange_weak(next, new_node,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                    break;
                }
            } else {
                tail_.compare_exchange_weak(tail, next,
                                           std::memory_order_release,
                                           std::memory_order_relaxed);
            }
        }
    }
    
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
                    return std::nullopt; // Empty
                }
                tail_.compare_exchange_weak(tail, next,
                                           std::memory_order_release,
                                           std::memory_order_relaxed);
            } else {
                uint32_t value = next->data.load(std::memory_order_relaxed);
                if (head_.compare_exchange_weak(head, next,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    delete head; // Safe: no other thread can access this node now
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

// WorkStealingQueue Implementation
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
        return false; // Full
    }
    
    buffer_[b & mask_].value.store(item, std::memory_order_relaxed);
    buffer_[b & mask_].valid.store(true, std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_release);
    return true;
}

std::optional<uint32_t> WorkStealingQueue::pop() {
    size_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t t = top_.load(std::memory_order_relaxed);
    
    if (t <= b) {
        uint32_t item = buffer_[b & mask_].value.load(std::memory_order_relaxed);
        if (t == b) {
            // Last item, race with thieves
            if (!top_.compare_exchange_strong(t, t + 1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                bottom_.store(b + 1, std::memory_order_relaxed);
                return std::nullopt;
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return item;
    } else {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

std::optional<uint32_t> WorkStealingQueue::steal() {
    size_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);
    
    if (t < b) {
        uint32_t item = buffer_[t & mask_].value.load(std::memory_order_relaxed);
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

}