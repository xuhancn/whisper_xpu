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

struct AudioDeviceInfo {
    int         index;       // PortAudio device index
    std::string name;        // human-readable name
    int         max_channels; // max input channels
    double      sample_rate; // default sample rate
    bool        is_default;  // true if this is the system default input
    std::string to_string() const;
};

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // List available input devices (structured)
    static std::vector<AudioDeviceInfo> enumerate_devices();

    // Start capturing from the specified input device.
    // Pass -1 for system default (does not change system setting).
    bool start(int device_id = -1, int sample_rate = 16000, int frames_per_buffer = 512);

    // Stop capturing
    void stop();

    // Set the callback to receive audio samples
    void set_callback(AudioCaptureCallback callback);

    // Check if currently capturing
    bool is_active() const;

    // Get the actual sample rate in use
    int sample_rate() const;

    // Get the device index in use
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
