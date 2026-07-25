#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

// Mic device index constants
static constexpr int kMicDefault = -1;  // system default input device

// Callback: provides PCM f32 16kHz mono audio samples.
// Called from the PortAudio callback thread.
using AudioCaptureCallback = std::function<size_t(const float* samples, size_t count)>;

struct AudioDeviceInfo {
    int         index;        // PortAudio device index
    std::string name;         // human-readable name
    int         max_channels; // max input channels
    double      sample_rate;  // default sample rate
    bool        is_default;   // true if this is the system default input
    std::string to_string() const;
};

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    // List available input devices (structured)
    static std::vector<AudioDeviceInfo> enumerate_devices();

    // Start capturing from the specified input device.
    // device_id: kMicDefault for system default, or a PortAudio device index.
    bool start(int device_id = kMicDefault, int sample_rate = 16000, int frames_per_buffer = 512);

    void stop();
    void set_callback(AudioCaptureCallback callback);
    bool is_active() const;
    int sample_rate() const;
    int device_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
