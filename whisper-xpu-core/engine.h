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

// Callback: fill buffer with up to max_samples PCM f32 16kHz audio samples.
// Returns the number of samples actually written (0 = end of stream).
using AudioSampleCallback = std::function<size_t(float* buffer, size_t max_samples)>;

class Engine {
public:
    // Load a whisper model.
    // model_path: path to a GGML model file (.bin or .ggml)
    // use_gpu: if true, attempt SYCL/GPU acceleration; falls back to CPU on failure
    Engine(const std::string& model_path, bool use_gpu = true);

    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Transcribe an entire audio file (WAV format).
    TranscriptionResult transcribe_file(const std::string& audio_path);

    // Transcribe streaming audio via callback.
    // callback is invoked repeatedly to get PCM f32 16kHz samples.
    TranscriptionResult transcribe_stream(AudioSampleCallback callback);

    // Returns true if GPU acceleration was successfully initialized
    bool is_gpu_enabled() const;

    // Returns a description of the compute device in use
    std::string device_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
