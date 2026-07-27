#pragma once

#include "export.h"

#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include <functional>

// whisper.cpp's per-inference state (forward decl — keeps whisper.h out of
// this public header).  Multiple whisper_state*s share ONE Engine's
// whisper_context so the parallel pool runs 4 concurrent inferences on a
// single read-only model copy (whisper.cpp issue #523).
struct whisper_state;

namespace whisper_xpu {

struct WHISPER_XPU_API TranscriptionResult {
    std::string text;
    double processing_time_ms;
    int segment_count;
    bool gpu_accelerated;
};

struct WHISPER_XPU_API BenchmarkResult {
    double total_audio_duration_s;
    double processing_time_ms;
    double realtime_factor;
    double rtf;
};

struct WHISPER_XPU_API VadConfig {
    bool   enabled               = false;
    float  max_speech_duration_s = 5.0f;
    int    speech_pad_ms         = 500;
    float  vad_threshold         = 0.5f;
    int    min_speech_duration_ms = 250;
    int    min_silence_duration_ms = 100;
    const char* vad_model_path   = nullptr;
};

// One transcribed segment with its chunk-local timestamp range (ms, relative
// to the start of the PCM passed to transcribe_window).  Used by the parallel
// scheduler's overlap merger: global time = chunk t_start + these offsets.
struct WHISPER_XPU_API ChunkSegment {
    std::string text;
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
};

// Result of transcribe_window: timestamped segments (chunk-local), the wall
// time spent, and whether the run was aborted via the abort flag.
struct WHISPER_XPU_API ChunkResult {
    std::vector<ChunkSegment> segments;
    double processing_time_ms = 0.0;
    bool aborted = false;
};

using AudioSampleCallback = std::function<size_t(float* buffer, size_t max_samples)>;

class WHISPER_XPU_API Engine {
public:
    // n_threads: 0 ⇒ std::thread::hardware_concurrency() (default, preserves
    // existing behavior).  A worker-pool caller passes core_count/pool_size so
    // the total across all workers ≈ core count (no oversubscription).
    Engine(const std::string& model_path, int device_id = 0, int n_threads = 0);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    TranscriptionResult transcribe_file(const std::string& audio_path,
                                        const VadConfig& vad = VadConfig{});

    // Stream transcription: repeatedly calls `callback` to pull PCM chunks
    // (16 kHz mono) and runs whisper_full on each, accumulating text.
    // If `abort_flag` is non-null and becomes true while a chunk is being
    // processed, the in-flight whisper_full is aborted (via its abort/
    // encoder-begin callbacks) and the pull loop exits promptly — so the
    // caller's stop() doesn't block on a slow CPU chunk.
    TranscriptionResult transcribe_stream(AudioSampleCallback callback,
                                         const std::atomic<bool>* abort_flag = nullptr);

    // Transcribe a single chunk of PCM (16 kHz mono).  Returns the chunk's
    // text (spaces trimmed).  If `abort_flag` is non-null and set true
    // mid-computation, whisper_full is aborted and "" is returned quickly.
    // Used by the real-time VAD chunker (AudioRecorder) so each detected
    // speech phrase can be transcribed and shown immediately.
    std::string transcribe_chunk(const float* pcm, int n_samples,
                                const std::atomic<bool>* abort_flag = nullptr);

    // ── Parallel-pool: shared-context inference ──
    //
    // The scheduler's 4 workers share ONE Engine (= one whisper_context =
    // one read-only model copy).  Each worker owns its own whisper_state
    // (KV cache / logits / decoders — all per-state), created here and freed
    // by free_state() after the worker thread joins.  This is whisper.cpp's
    // "one context + N states" pattern (issue #523) and is what keeps pool
    // RAM at ~1× model + 4 small states instead of 4× model.
    //
    // Caller owns the returned state.  Lifetime: create_state after Engine
    // init; free_state after the worker thread that used it has joined (the
    // state must not be in use during whisper_full_with_state when freed).
    whisper_state* create_state();
    void free_state(whisper_state* st);

    // Transcribe a window using a caller-owned per-worker `st` (from
    // create_state), running whisper_full_with_state(ctx, st, …) so many
    // states can share one context concurrently.
    //
    // `detected_language` is a PER-WORKER cache owned by the caller: empty on
    // the first call ⇒ language="auto" (auto-detect via its own encoder pass)
    // and the result is written back; subsequent calls pin it, skipping the
    // detection pass (halves per-window encoder cost — same optimization the
    // old per-Engine cache had, moved to per-worker because the context is now
    // shared).  `n_threads` overrides the Engine's default (a pool passes
    // cores/pool_size so 4×n_threads ≈ core count).
    //
    // Emits multiple timestamped segments (single_segment=false,
    // no_timestamps=false) — required for the merger's overlap dedup by
    // midpoint.  abort_flag aborts mid-computation (polled before each ggml
    // op + encoder_begin gate), same wiring as transcribe_chunk.
    ChunkResult transcribe_window_with_state(whisper_state* st, int n_threads,
                                             std::string& detected_language,
                                             const float* pcm, int n_samples,
                                             const std::atomic<bool>* abort_flag = nullptr);

    BenchmarkResult benchmark(const std::string& audio_path, const VadConfig& vad);

    bool is_gpu_enabled() const;
    std::string device_description() const;
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
