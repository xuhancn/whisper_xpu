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

using AudioSampleCallback = std::function<size_t(float* buffer, size_t max_samples)>;

class Engine {
public:
    // model_path: path to a GGML model file (.bin or .ggml)
    // device_id: kDeviceCPU (CPU) or 0+ (GPU index from get_available_devices())
    Engine(const std::string& model_path, int device_id = 0);

    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    TranscriptionResult transcribe_file(const std::string& audio_path);
    TranscriptionResult transcribe_stream(AudioSampleCallback callback);

    bool is_gpu_enabled() const;
    std::string device_description() const;
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace whisper_xpu
