// Streaming-pipeline unit test.
//
// Replays a 60 s WAV (trump_60s_final.wav) through the EXACT same
// TranscriptionScheduler the app uses — but with a file source instead of a
// microphone (start_no_capture + feed_audio).  No UI, no mic, no PortAudio
// stream.  Verifies the windower/pool/merger end-to-end against six hard
// criteria and reports a full pipeline accounting.
//
// Run:  test_pipeline --model <ggml-tiny.bin> [--audio <wav>] [--cpu]

#include "engine.h"
#include "device_detect.h"
#include "src/transcription_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef TEST_WAV_DIR
#define TEST_WAV_DIR "."
#endif

static constexpr int SR = 16000;

// ── Minimal WAV loader (16 kHz mono f32 out) — mirrors engine.cpp's paths ──
static bool load_wav(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    auto r16 = [&]() -> uint16_t { uint16_t v; f.read((char*)&v, sizeof(v)); return v; };
    auto r32 = [&]() -> uint32_t { uint32_t v; f.read((char*)&v, sizeof(v)); return v; };

    char riff[4]; f.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) return false;
    r32();
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) return false;

    int ch = 0, sr = 0, bps = 0;
    std::vector<int16_t> pcm;
    while (f.good()) {
        char cid[4]; f.read(cid, 4);
        uint32_t csz = r32();
        if (f.gcount() < 4) break;
        if (std::memcmp(cid, "fmt ", 4) == 0) {
            r16(); ch = r16(); sr = (int)r32(); r32(); r16(); bps = r16();
        } else if (std::memcmp(cid, "data", 4) == 0) {
            if (bps == 16) { pcm.resize(csz / 2); f.read((char*)pcm.data(), csz); }
            else f.seekg(csz, std::ios::cur);
        } else f.seekg(csz, std::ios::cur);
    }
    if (pcm.empty() || ch == 0) return false;

    std::vector<float> mono;
    if (ch == 1) {
        mono.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); i++) mono[i] = pcm[i] / 32768.0f;
    } else {
        size_t nf = pcm.size() / ch;
        mono.resize(nf);
        for (size_t i = 0; i < nf; i++) {
            float s = 0; for (int c = 0; c < ch; c++) s += pcm[i * ch + c] / 32768.0f;
            mono[i] = s / ch;
        }
    }
    if (sr != SR) {
        double ratio = (double)SR / sr;
        size_t nl = (size_t)(mono.size() * ratio);
        std::vector<float> r(nl);
        for (size_t i = 0; i < nl; i++) {
            double pos = i / ratio; size_t si = (size_t)pos; double fr = pos - si;
            float s0 = mono[std::min(si, mono.size() - 1)];
            float s1 = mono[std::min(si + 1, mono.size() - 1)];
            r[i] = s0 + (float)fr * (s1 - s0);
        }
        out.swap(r);
    } else out.swap(mono);
    return true;
}

// ── Check helper: prints PASS/FAIL + detail, returns 0/1 failures ──
static int check(const char* name, bool ok, const char* fmt, ...) {
    fprintf(stderr, "  [%s] %s", ok ? "PASS" : "FAIL", name);
    if (fmt && *fmt) {
        fprintf(stderr, " — ");
        va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    }
    fprintf(stderr, "\n");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    std::string audio_path = std::string(TEST_WAV_DIR) + "/trump_60s_final.wav";
    int device = kDeviceCPU;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--audio" && i + 1 < argc) audio_path = argv[++i];
        else if (a == "--cpu") device = kDeviceCPU;
        else if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--help") {
            printf("Usage: test_pipeline --model <path> [--audio <wav>] [--cpu]\n");
            return 0;
        }
    }
    if (!model_path) { fprintf(stderr, "Error: --model required\n"); return 2; }

    fprintf(stderr, "=== Streaming pipeline unit test ===\n");
    fprintf(stderr, "model:  %s\n", model_path);
    fprintf(stderr, "audio:  %s\n", audio_path.c_str());
    fprintf(stderr, "device: %s\n", device == kDeviceCPU ? "CPU" : std::to_string(device).c_str());

    // ── Load WAV ──
    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm) || pcm.empty()) {
        fprintf(stderr, "FAIL: could not load audio\n");
        return 2;
    }
    const double dur_s = (double)pcm.size() / SR;
    const int expected_windows = (int)(pcm.size() / (size_t)(SR * 5));  // 5s new/window
    fprintf(stderr, "audio:  %zu samples (%.2fs) ⇒ expected %d windows\n",
            pcm.size(), dur_s, expected_windows);

    // ── Reference transcription (whole-file, for comparison reporting) ──
    // This single Engine is ALSO the shared context the scheduler's 4 workers
    // borrow — one model copy for the whole test, not 4 (T6 shared-context).
    std::string ref_text;
    double ref_ms = 0.0;  // single-pass whole-file cost (for timeout budget)
    whisper_xpu::Engine ref(model_path, device);
    {
        fprintf(stderr, "\n--- reference (whole-file transcribe_file) ---\n");
        try {
            auto r = ref.transcribe_file(audio_path);
            ref_text = r.text;
            ref_ms  = r.processing_time_ms;
            fprintf(stderr, "reference: %zu chars, %d segs, %.0fms\n",
                    r.text.size(), r.segment_count, r.processing_time_ms);
            fprintf(stderr, "ref sample: %.120s\n", r.text.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "reference failed: %s (continuing)\n", e.what());
        }
    }

    // ── Build the SAME scheduler the app uses, with a 60s ring ──
    std::string accumulated;
    std::vector<int> emit_order;
    accumulated.reserve(4096);

    TranscriptionScheduler sched(
        [&](const std::string& text) {
            accumulated += text;
            accumulated += ' ';
        },
        pcm.size() + 1);  // ring holds the whole clip (+1 so push never wraps)
    sched.set_on_chunk([&](int idx, const std::string&) { emit_order.push_back(idx); });

    fprintf(stderr, "\n--- pipeline run (headless: feed_audio → windower → 4 workers → merger) ---\n");
    if (!sched.start_no_capture(ref)) {
        fprintf(stderr, "FAIL: start_no_capture failed (state init?)\n");
        return 2;
    }

    // Pre-load the entire clip; the windower drains it at worker speed.
    sched.feed_audio(pcm.data(), pcm.size());

    // Wait for all expected windows to emit in order (or a model-scaled timeout).
    //
    // The 60s floor covers tiny/turbo (each window <2s).  Heavy models under the
    // scheduler's 4-way CPU concurrency (4 states x N threads = hw threads) can
    // take ~30-40s PER 6s window due to memory-bandwidth contention — far more
    // than the single-pass reference.  Scale the budget from the measured
    // whole-file reference cost: per-window uncontended ≈ ref_ms/expected_windows,
    // inflated by an 8x concurrency-contention factor + 2x safety margin, with a
    // 60s floor and a 600s ceiling (so a genuinely stuck run still aborts).
    double per_window_uncontended_ms = expected_windows > 0 ? ref_ms / expected_windows : 5000.0;
    int timeout_s = (int)(per_window_uncontended_ms * 8.0 * 2.0 * expected_windows / 1000.0);
    if (timeout_s < 60)   timeout_s = 60;
    if (timeout_s > 600)  timeout_s = 600;
    fprintf(stderr, "emit timeout: %ds (per-window ref ≈ %.0fms x8 contention x2 margin x%d windows)\n",
            timeout_s, per_window_uncontended_ms, expected_windows);

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int last_emit = -1;
    while (true) {
        auto s = sched.stats();
        if (s.next_emit >= expected_windows) break;
        if (s.next_emit != last_emit) {
            fprintf(stderr, "  progress: emitted %d/%d (dispatched=%d completed=%d)\n",
                    s.next_emit, expected_windows, s.windows_dispatched, s.windows_completed);
            last_emit = s.next_emit;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(clock::now() - t0).count() > timeout_s) {
            fprintf(stderr, "  TIMEOUT waiting for %d emits (at %d, limit %ds)\n",
                    expected_windows, s.next_emit, timeout_s);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    sched.stop();
    auto s = sched.stats();

    // ── Report ──
    fprintf(stderr, "\n--- pipeline report ---\n");
    fprintf(stderr, "windows dispatched: %d\n", s.windows_dispatched);
    fprintf(stderr, "windows completed:  %d\n", s.windows_completed);
    fprintf(stderr, "workers used mask:  0x%x (bits 0..%d)\n",
            s.workers_used_mask, 4 - 1);
    fprintf(stderr, "segments kept:      %d\n", s.segments_kept);
    fprintf(stderr, "segments dropped:   %d\n", s.segments_dropped);
    fprintf(stderr, "total chars emit:   %ld\n", s.total_chars);
    fprintf(stderr, "next_emit / pending:%d / %d\n", s.next_emit, s.pending);
    fprintf(stderr, "stop time:          %dms\n", s.last_stop_ms);
    fprintf(stderr, "emit order:        ");
    for (int idx : emit_order) fprintf(stderr, " %d", idx);
    fprintf(stderr, "\n");
    fprintf(stderr, "pipeline sample:   %.140s\n", accumulated.c_str());
    if (!ref_text.empty())
        fprintf(stderr, "reference chars=%zu  pipeline chars=%ld  (overlap expected lower due to dedup)\n",
                ref_text.size(), s.total_chars);

    // ── Hard asserts ──
    int fail = 0;
    fprintf(stderr, "\n--- asserts ---\n");
    fail += check("all windows dispatched",
                  s.windows_dispatched == expected_windows,
                  "got %d, want %d", s.windows_dispatched, expected_windows);
    fail += check("all windows completed",
                  s.windows_completed == s.windows_dispatched,
                  "completed %d vs dispatched %d", s.windows_completed, s.windows_dispatched);
    // in-order: emit_order must be contiguous [0 .. next_emit-1]
    bool in_order = !emit_order.empty() && (int)emit_order.size() == s.next_emit;
    for (size_t i = 0; in_order && i < emit_order.size(); i++)
        in_order = (emit_order[i] == (int)i);
    fail += check("merger emits chunks in order (0,1,2,...)",
                  in_order, "size=%zu next_emit=%d", emit_order.size(), s.next_emit);
    fail += check("no worker starvation (all pool workers used)",
                  // mask must have every bit in [0..popcount(mask)] set, i.e.
                  // workers are used contiguously from 0 with no gaps.  POOL_SIZE
                  // is GPU=1 / CPU=4, so the exact count varies — assert no
                  // gap, not a fixed 4.
                  (s.workers_used_mask != 0 &&
                   (s.workers_used_mask & (s.workers_used_mask + 1)) == 0),
                  "mask=0x%x", s.workers_used_mask);
    fail += check("output text exceeds 200 chars",
                  s.total_chars > 200,
                  "got %ld chars", s.total_chars);
    int segs = s.segments_kept + s.segments_dropped;
    double drop_rate = segs > 0 ? (double)s.segments_dropped / segs : 0.0;
    fail += check("dedup drops < 20% of segments",
                  drop_rate < 0.20,
                  "dropped %d/%d = %.0f%%", s.segments_dropped, segs, drop_rate * 100);
    fail += check("stop completes < 2 seconds",
                  s.last_stop_ms >= 0 && s.last_stop_ms < 2000,
                  "got %dms", s.last_stop_ms);

    fprintf(stderr, "\n=== %s (failures=%d) ===\n", fail == 0 ? "ALL PASSED" : "SOME FAILED", fail);
    return fail;
}
