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
    // device_id: -1 = CPU, 0+ = GPU device index from get_available_devices()
    Engine(const std::string& model_path, int device_id = 0);

    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Transcribe an entire audio file (WAV format).
    TranscriptionResult transcribe_file(const std::string& audio_path);

    // Transcribe streaming audio via callback.
    TranscriptionResult transcribe_stream(AudioSampleCallback callback);

    // Returns true if GPU acceleration was successfully initialized
    bool is_gpu_enabled() const;

    // Returns a description of the compute device in use
    std::string device_description() const;

    // Returns the device index in use (-1 for CPU)
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
