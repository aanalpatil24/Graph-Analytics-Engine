// tests/test_concurrent_queue.cpp
#include <gtest/gtest.h>
#include "graph/concurrent_queue.hpp"
#include <thread>
#include <vector>
#include <atomic>

using namespace graph::concurrent;

TEST(ConcurrentQueueTest, SingleThreadOperations) {
    LockFreeQueue queue;
    
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
    
    queue.push(42);
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 1);
    
    auto val = queue.pop();
    EXPECT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
    EXPECT_TRUE(queue.empty());
}

TEST(ConcurrentQueueTest, MultiThreadPushPop) {
    LockFreeQueue queue;
    std::atomic<size_t> pushed{0};
    std::atomic<size_t> popped{0};
    const int ITERATIONS = 10000;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    // 2 producers
    for (int t = 0; t < 2; ++t) {
        producers.emplace_back([&]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                queue.push(i);
                pushed.fetch_add(1);
            }
        });
    }
    
    // 2 consumers
    for (int t = 0; t < 2; ++t) {
        consumers.emplace_back([&]() {
            int local_popped = 0;
            while (local_popped < ITERATIONS) {
                auto val = queue.pop();
                if (val) {
                    popped.fetch_add(1);
                    ++local_popped;
                }
            }
        });
    }
    
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();
    
    EXPECT_EQ(pushed.load(), ITERATIONS * 2);
    EXPECT_EQ(popped.load(), ITERATIONS * 2);
    EXPECT_TRUE(queue.empty());
}

TEST(WorkStealingQueueTest, OwnerOperations) {
    WorkStealingQueue queue(1024);
    
    EXPECT_TRUE(queue.empty());
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(queue.push(i));
    }
    
    for (int i = 99; i >= 0; --i) {
        auto val = queue.pop();
        EXPECT_TRUE(val.has_value());
        EXPECT_EQ(*val, i);
    }
    
    EXPECT_TRUE(queue.empty());
}