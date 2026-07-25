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
    double total_audio_duration_s;
    double processing_time_ms;
    double realtime_factor;
    double rtf;
};

struct VadConfig {
    bool   enabled               = false;
    float  max_speech_duration_s = 5.0f;
    int    speech_pad_ms         = 500;
    float  vad_threshold         = 0.5f;
    int    min_speech_duration_ms = 250;
    int    min_silence_duration_ms = 100;
};

using AudioSampleCallback = std::function<size_t(float* buffer, size_t max_samples)>;

// Merge transcribed text segments, deduplicating overlapping suffixes.
std::string merge_segments(const std::vector<const char*>& segments);
std::string merge_segments(const std::vector<std::string>& segments);

class Engine {
public:
    Engine(const std::string& model_path, int device_id = 0);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    TranscriptionResult transcribe_file(const std::string& audio_path,
                                        const VadConfig& vad = VadConfig{});

    TranscriptionResult transcribe_stream(AudioSampleCallback callback);

    BenchmarkResult benchmark(const std::string& audio_path, const VadConfig& vad);

    bool is_gpu_enabled() const;
    std::string device_description() const;
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
