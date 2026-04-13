// include/utils/aligned_allocator.hpp
#pragma once
#include <memory>
#include <cstdlib>
#include "graph/types.hpp"

namespace graph {

template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return aligned_alloc<T>(n);
    }

    void deallocate(pointer p, size_type) noexcept {
        aligned_free(p);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template<typename T>
using CacheAlignedVector = std::vector<T, AlignedAllocator<T>>;

} // namespace graph