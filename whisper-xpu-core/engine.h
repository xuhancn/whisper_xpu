#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace whisper_xpu {

struct TranscriptionResult {
    std::string text;
    double processing_time_ms;
    int segment_count;
    bool gpu_accelerated;
};

struct BenchmarkResult {
    double total_audio_duration_s;   // actual audio length processed
    double processing_time_ms;       // wall-clock time
    double realtime_factor;          // processing_time_ms / (total_audio_duration_s * 1000)
    double rtf;                      // shorthand
};

struct VadConfig {
    bool   enabled               = false;
    float  max_speech_duration_s = 5.0f;   // max seconds per VAD segment
    int    speech_pad_ms         = 500;    // padding around speech (1s total overlap)
    float  vad_threshold         = 0.5f;   // VAD probability threshold
    int    min_speech_duration_ms = 250;   // min valid speech duration
    int    min_silence_duration_ms = 100;  // min silence to split
};

using AudioSampleCallback = std::function<size_t(float* buffer, size_t max_samples)>;

class Engine {
public:
    Engine(const std::string& model_path, int device_id = 0);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Transcribe a full audio file.
    TranscriptionResult transcribe_file(const std::string& audio_path,
                                        const VadConfig& vad = VadConfig{});

    // Transcribe streaming audio via callback.
    TranscriptionResult transcribe_stream(AudioSampleCallback callback);

    // Run benchmark on an audio file with given VAD config.
    BenchmarkResult benchmark(const std::string& audio_path, const VadConfig& vad);

    bool is_gpu_enabled() const;
    std::string device_description() const;
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
