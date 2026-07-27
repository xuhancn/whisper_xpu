#include <gtest/gtest.h>
#include "src/ring_buffer.h"
#include <thread>
#include <atomic>

// ── push / pull basic ──
TEST(RingBuffer, PushPullBasic) {
    RingBuffer<float> rb(100);
    float in[50], out[50] = {};
    for (int i = 0; i < 50; ++i) in[i] = (float)i;

    rb.push(in, 50);
    EXPECT_EQ(rb.available(), 50u);

    size_t n = rb.pull(out, 50);
    EXPECT_EQ(n, 50u);
    for (int i = 0; i < 50; ++i) EXPECT_FLOAT_EQ(out[i], (float)i);
    EXPECT_EQ(rb.available(), 0u);
}

// ── wraparound: push past capacity overwrites oldest ──
TEST(RingBuffer, Wraparound) {
    RingBuffer<int> rb(10);
    int in[15];
    for (int i = 0; i < 15; ++i) in[i] = i;

    rb.push(in, 15);  // only last 10 should remain: 5..14
    EXPECT_EQ(rb.available(), 10u);

    int out[10] = {};
    size_t n = rb.pull(out, 10);
    EXPECT_EQ(n, 10u);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(out[i], 5 + i);
}

// ── empty pull returns 0 ──
TEST(RingBuffer, EmptyPull) {
    RingBuffer<float> rb(50);
    float out[10];
    EXPECT_EQ(rb.pull(out, 10), 0u);
}

// ── partial pull: pull less than available ──
TEST(RingBuffer, PartialPull) {
    RingBuffer<int> rb(20);
    int in[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    rb.push(in, 8);
    EXPECT_EQ(rb.available(), 8u);

    int out[3] = {};
    EXPECT_EQ(rb.pull(out, 3), 3u);
    EXPECT_EQ(out[0], 10); EXPECT_EQ(out[1], 20); EXPECT_EQ(out[2], 30);
    EXPECT_EQ(rb.available(), 5u);

    int out2[5] = {};
    EXPECT_EQ(rb.pull(out2, 5), 5u);
    EXPECT_EQ(out2[0], 40); EXPECT_EQ(out2[4], 80);
    EXPECT_EQ(rb.available(), 0u);
}

// ── available reflects correct count ──
TEST(RingBuffer, Available) {
    RingBuffer<int> rb(100);
    EXPECT_EQ(rb.available(), 0u);
    int d = 42;
    rb.push(&d, 1);
    EXPECT_EQ(rb.available(), 1u);
    rb.pull(&d, 1);
    EXPECT_EQ(rb.available(), 0u);
}

// ── clear ──
TEST(RingBuffer, Clear) {
    RingBuffer<int> rb(10);
    int d[5] = {1,2,3,4,5};
    rb.push(d, 5);
    EXPECT_EQ(rb.available(), 5u);
    rb.clear();
    EXPECT_EQ(rb.available(), 0u);
    int out[1];
    EXPECT_EQ(rb.pull(out, 1), 0u);
}

// ── thread safety: concurrent push from audio thread, pull from worker ──
TEST(RingBuffer, ThreadSafety) {
    RingBuffer<float> rb(160000);  // 10s @ 16kHz
    std::atomic<bool> done{false};
    std::atomic<size_t> total_pushed{0};
    std::atomic<size_t> total_pulled{0};
    std::atomic<size_t> rounds{0};

    // Producer: 500 pushes of 512 samples = 256k total (exceeds capacity)
    // Oldest 96k will be overwritten — ring buffer semantics.
    std::thread producer([&]() {
        float buf[512];
        for (int r = 0; r < 500; ++r) {
            for (int i = 0; i < 512; ++i) buf[i] = (float)(r * 512 + i);
            rb.push(buf, 512);
            total_pushed += 512;
        }
        done = true;
    });

    // Consumer: pulls in chunks (with deliberate slowness)
    std::thread consumer([&]() {
        float buf[1024];
        while (!done || rb.available() > 0) {
            size_t n = rb.pull(buf, 1024);
            total_pulled += n;
            if (n > 0) ++rounds;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        // Flush remainder
        while (rb.available() > 0) {
            size_t n = rb.pull(buf, 1024);
            total_pulled += n;
        }
    });

    producer.join();
    consumer.join();

    // Consumer pulled everything that was not overwritten.
    // With 160k capacity and 256k pushed, the first 96k are gone.
    EXPECT_GT(total_pulled.load(), 0u);
    EXPECT_LE(total_pulled.load(), total_pushed.load());
    EXPECT_GE(total_pulled.load(), rb.capacity());  // at least capacity worth survived
    EXPECT_GT(rounds.load(), 0u);
    EXPECT_EQ(rb.available(), 0u);  // drained
}
