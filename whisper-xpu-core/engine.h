#pragma once

#include "export.h"

#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include <functional>

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

    // Transcribe a window of PCM (16 kHz mono) and return its segments with
    // chunk-local timestamps (t0_ms/t1_ms, relative to `pcm` start).  The
    // parallel scheduler adds the chunk's global t_start and the merger
    // dedupes across the 1s overlap by midpoint.
    //
    // Unlike transcribe_chunk this emits multiple timestamped segments
    // (single_segment=false, no_timestamps=false) — required for overlap
    // merging.  Language is detected on the first call and pinned thereafter
    // (per-Engine cache).  abort_flag aborts mid-computation.
    ChunkResult transcribe_window(const float* pcm, int n_samples,
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
