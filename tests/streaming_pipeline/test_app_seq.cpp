// App-sequence repro test.
//
// Reproduces the EXACT start sequence whisper_xpu_app uses when Record is
// clicked — Engine ctor + init_states (4 GPU states) + 4 worker threads
// issuing transcribe_window_with_state as the FIRST GPU compute in the
// process — with NO prior transcribe_file() warmup on the main thread.
//
// test_pipeline.cpp differs from the app in exactly one way: it calls
// ref.transcribe_file(audio) on the main thread BEFORE start_no_capture, so
// the SYCL runtime/device-image registration is exercised once before any
// worker touches the GPU.  The app does no such warmup.  This test removes
// the warmup to see if the app's sycl8.dll 0xc0000005 crash reproduces.
//
// Run:  test_app_seq --model <ggml-tiny.bin> [--audio <wav>] [--device <n>]

#include "engine.h"
#include "device_detect.h"
#include "src/transcription_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef TEST_WAV_DIR
#define TEST_WAV_DIR "."
#endif

static constexpr int SR = 16000;

static bool load_wav(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[4]; f.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) return false;
    f.ignore(4);  // size
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) return false;
    int sampleRate = 0, bits = 0, channels = 1;
    while (f) {
        char id[4]; f.read(id, 4);
        unsigned sz = 0;
        f.read(reinterpret_cast<char*>(&sz), 4);
        if (!f) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            short fmt; f.read(reinterpret_cast<char*>(&fmt), 2);
            short ch;   f.read(reinterpret_cast<char*>(&ch), 2);
            f.read(reinterpret_cast<char*>(&sampleRate), 4);
            f.ignore(4);  // byte rate
            f.ignore(2);  // block align
            f.read(reinterpret_cast<char*>(&bits), 2);
            if (sz > 16) f.ignore(sz - 16);
            channels = ch;
        } else if (std::memcmp(id, "data", 4) == 0) {
            size_t n = sz / (bits / 8);
            out.resize(n);
            if (bits == 16) {
                std::vector<short> raw(n);
                f.read(reinterpret_cast<char*>(raw.data()), sz);
                for (size_t i = 0; i < n; ++i)
                    out[i] = raw[i] / 32768.0f;
            } else if (bits == 32) {
                f.read(reinterpret_cast<char*>(out.data()), sz);
            } else {
                return false;
            }
            if (channels > 1) {
                size_t mono = n / channels;
                std::vector<float> m(mono, 0.0f);
                for (size_t i = 0; i < mono; ++i)
                    for (int c = 0; c < channels; ++c)
                        m[i] += out[i * channels + c] / channels;
                out.swap(m);
            }
            break;
        } else {
            f.ignore(sz);
        }
        if (sz & 1) f.ignore(1);
    }
    if (sampleRate != SR) {
        // crude linear resample to 16k
        double step = (double)sampleRate / SR;
        size_t outN = (size_t)(out.size() / step);
        std::vector<float> r(outN);
        for (size_t i = 0; i < outN; ++i) {
            double p = i * step;
            size_t lo = (size_t)p;
            double frac = p - lo;
            float a = (lo < out.size()) ? out[lo] : 0.0f;
            float b = (lo + 1 < out.size()) ? out[lo + 1] : a;
            r[i] = (float)(a + (b - a) * frac);
        }
        out.swap(r);
    }
    return !out.empty();
}

int main(int argc, char** argv) {
    std::string model_path, audio_path = std::string(TEST_WAV_DIR) + "/trump_60s_final.wav";
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--audio" && i + 1 < argc) audio_path = argv[++i];
        else if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--cpu") device = -1;
        else {
            fprintf(stderr, "Usage: test_app_seq --model <path> [--audio <wav>] [--device <n>] [--cpu]\n");
            return 2;
        }
    }
    if (model_path.empty()) { fprintf(stderr, "Error: --model required\n"); return 2; }

    fprintf(stderr, "=== App-sequence repro (NO transcribe_file warmup) ===\n");
    fprintf(stderr, "model:  %s\n", model_path.c_str());
    fprintf(stderr, "audio:  %s\n", audio_path.c_str());
    fprintf(stderr, "device: %s\n", device < 0 ? "CPU" : std::to_string(device).c_str());

    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm) || pcm.empty()) {
        fprintf(stderr, "FAIL: could not load audio\n"); return 2;
    }
    // Feed only the first ~10s so the test is quick but still drives >=1 window.
    if (pcm.size() > (size_t)(SR * 10)) pcm.resize(SR * 10);
    const int expected_windows = (int)(pcm.size() / (size_t)(SR * 5));
    fprintf(stderr, "audio: %zu samples, expected %d windows\n", pcm.size(), expected_windows);

    // ── Engine ctor — NO transcribe_file warmup, exactly like the app ──
    setvbuf(stderr, nullptr, _IONBF, 0);  // unbuffered: don't lose crash lines
    fprintf(stderr, "constructing Engine (device %d)...\n", device);
    whisper_xpu::Engine eng(model_path, device);
    fprintf(stderr, "Engine constructed OK\n");

    // ── Scheduler: start_no_capture + feed, workers are first GPU callers ──
    std::string acc;
    TranscriptionScheduler sched(
        [&](const std::string& text) { acc += text; acc += ' '; },
        pcm.size() + 1);
    sched.set_on_chunk([&](int, const std::string& t) {
        fprintf(stderr, "  chunk: '%.60s'\n", t.c_str());
    });

    fprintf(stderr, "start_no_capture (workers will be FIRST GPU callers)...\n");
    if (!sched.start_no_capture(eng)) {
        fprintf(stderr, "FAIL: start_no_capture failed\n"); return 2;
    }
    sched.feed_audio(pcm.data(), pcm.size());

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int last = -1;
    while (true) {
        auto s = sched.stats();
        if (s.next_emit >= expected_windows) break;
        if (s.next_emit != last) {
            fprintf(stderr, "  progress: emitted %d/%d (dispatched=%d completed=%d)\n",
                    s.next_emit, expected_windows, s.windows_dispatched, s.windows_completed);
            last = s.next_emit;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(clock::now() - t0).count() > 90) {
            fprintf(stderr, "  TIMEOUT at %d/%d\n", s.next_emit, expected_windows);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "calling stop...\n");
    sched.stop();
    fprintf(stderr, "acc=%zu chars\n", acc.size());
    fprintf(stderr, "sample: %.120s\n", acc.c_str());
    fprintf(stderr, "RESULT: %s\n", acc.empty() ? "EMPTY (crash/no output)" : "GOT TEXT");
    return acc.empty() ? 1 : 0;
}
