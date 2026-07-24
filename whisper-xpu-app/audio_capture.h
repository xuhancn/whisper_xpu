#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

// Callback: provides PCM f32 16kHz mono audio samples.
// Called from the PortAudio callback thread.
// Returns the number of samples actually consumed.
using AudioCaptureCallback = std::function<size_t(const float* samples, size_t count)>;

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // List available input devices (human-readable)
    static std::vector<std::string> enumerate_devices();

    // Start capturing from the default input device.
    // sample_rate: 16000 Hz recommended (what whisper.cpp expects)
    // frames_per_buffer: ~512 (~32ms at 16kHz)
    bool start(int sample_rate = 16000, int frames_per_buffer = 512);

    // Stop capturing
    void stop();

    // Set the callback to receive audio samples
    void set_callback(AudioCaptureCallback callback);

    // Check if currently capturing
    bool is_active() const;

    // Get the actual sample rate in use
    int sample_rate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
