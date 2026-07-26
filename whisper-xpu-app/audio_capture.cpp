#include "audio_capture.h"
#include "portaudio.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>

// ---------------------------------------------------------------------------
// AudioDeviceInfo
// ---------------------------------------------------------------------------

std::string AudioDeviceInfo::to_string() const {
    std::ostringstream oss;
    oss << name;
    if (max_channels > 0)
        oss << " (" << max_channels << "ch, " << (int)sample_rate << " Hz)";
    if (is_default)
        oss << " [default]";
    return oss.str();
}

// ---------------------------------------------------------------------------
// AudioCapture::Impl
// ---------------------------------------------------------------------------

struct AudioCapture::Impl {
    PaStream* stream = nullptr;
    bool active = false;
    int sampleRate = 16000;
    int deviceId = kMicDefault;
    AudioCaptureCallback callback;

    static int paCallback(const void* input, void* output,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void* userData) {
        auto* impl = static_cast<Impl*>(userData);
        if (!impl || !impl->callback || !input) return paContinue;

        const float* samples = static_cast<const float*>(input);
        impl->callback(samples, frameCount);

        (void)output;
        (void)timeInfo;
        (void)statusFlags;
        return paContinue;
    }
};

AudioCapture::AudioCapture() : pimpl_(std::make_unique<Impl>()) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "[audio] PortAudio init error: %s\n", Pa_GetErrorText(err));
    }
}

AudioCapture::~AudioCapture() {
    stop();
    Pa_Terminate();
}

// ---------------------------------------------------------------------------
// Enumerate input devices
// ---------------------------------------------------------------------------

std::vector<AudioDeviceInfo> AudioCapture::enumerate_devices() {
    std::vector<AudioDeviceInfo> list;

    PaError err = Pa_Initialize();
    bool need_term = (err == paNoError);

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        if (need_term) Pa_Terminate();
        return list;
    }

    int default_dev = Pa_GetDefaultInputDevice();

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info || info->maxInputChannels < 1)
            continue;

        // Skip devices with empty or whitespace-only names
        std::string name = info->name ? info->name : "";
        if (name.empty() || name.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        AudioDeviceInfo dev;
        dev.index        = i;
        dev.name         = name;
        dev.max_channels = info->maxInputChannels;
        dev.sample_rate  = info->defaultSampleRate;
        dev.is_default   = (i == default_dev);
        list.push_back(dev);
    }

    if (need_term) Pa_Terminate();
    return list;
}

// ---------------------------------------------------------------------------
// Start / Stop
// ---------------------------------------------------------------------------

bool AudioCapture::start(int device_id, int sample_rate, int frames_per_buffer) {
    if (pimpl_->active) return true;

    // kMicDefault = use system default; concrete index = use that device
    int device = device_id;
    if (device == kMicDefault) {
        device = Pa_GetDefaultInputDevice();
        if (device == paNoDevice) {
            fprintf(stderr, "[audio] No default input device found\n");
            return false;
        }
    }

    // Validate device exists and has input channels
    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(device);
    if (!devInfo) {
        fprintf(stderr, "[audio] Device %d not found\n", device);
        return false;
    }
    if (devInfo->maxInputChannels < 1) {
        fprintf(stderr, "[audio] Device %d has no input channels\n", device);
        return false;
    }

    PaStreamParameters inputParams;
    memset(&inputParams, 0, sizeof(inputParams));
    inputParams.device = device;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(
        &pimpl_->stream, &inputParams, nullptr,
        sample_rate, frames_per_buffer, paClipOff,
        Impl::paCallback, pimpl_.get());

    if (err != paNoError) {
        fprintf(stderr, "[audio] OpenStream error: %s\n", Pa_GetErrorText(err));
        return false;
    }

    err = Pa_StartStream(pimpl_->stream);
    if (err != paNoError) {
        fprintf(stderr, "[audio] StartStream error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(pimpl_->stream);
        pimpl_->stream = nullptr;
        return false;
    }

    pimpl_->sampleRate = sample_rate;
    pimpl_->deviceId   = device;
    pimpl_->active     = true;
    fprintf(stderr, "[audio] Capture started (device %d: %s, %d Hz)\n",
            device, devInfo->name, sample_rate);
    return true;
}

void AudioCapture::stop() {
    if (!pimpl_->active) return;
    if (pimpl_->stream) {
        Pa_StopStream(pimpl_->stream);
        Pa_CloseStream(pimpl_->stream);
        pimpl_->stream = nullptr;
    }
    pimpl_->active = false;
    fprintf(stderr, "[audio] Capture stopped\n");
}

void AudioCapture::set_callback(AudioCaptureCallback cb) {
    pimpl_->callback = std::move(cb);
}

bool AudioCapture::is_active() const { return pimpl_->active; }
int AudioCapture::sample_rate() const { return pimpl_->sampleRate; }
int AudioCapture::device_id() const { return pimpl_->deviceId; }
