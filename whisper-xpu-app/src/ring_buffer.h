#pragma once

#include <vector>
#include <mutex>
#include <algorithm>

// Thread-safe fixed-capacity ring buffer.
// push() overwrites oldest data when full.
// pull() returns up to max_n samples, returns actual count.
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : m_buf(capacity) {}

    // Write n samples; oldest are overwritten if buffer overflows
    void push(const T* data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (size_t i = 0; i < n; ++i) {
            m_buf[m_write] = data[i];
            m_write = (m_write + 1) % m_buf.size();
            if (m_count < m_buf.size())
                ++m_count;
            else
                m_read = (m_read + 1) % m_buf.size();  // advance read past oldest
        }
    }

    // Pull up to max_n samples into out[]. Returns number actually pulled.
    size_t pull(T* out, size_t max_n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t n = std::min(max_n, m_count);
        for (size_t i = 0; i < n; ++i) {
            out[i] = m_buf[m_read];
            m_read = (m_read + 1) % m_buf.size();
        }
        m_count -= n;
        return n;
    }

    // Number of unread samples currently in the buffer
    size_t available() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_count;
    }

    // Reset to empty
    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_write = 0;
        m_read  = 0;
        m_count = 0;
    }

    size_t capacity() const { return m_buf.size(); }

private:
    mutable std::mutex m_mutex;
    std::vector<T>     m_buf;
    size_t             m_write = 0;
    size_t             m_read  = 0;
    size_t             m_count = 0;
};
