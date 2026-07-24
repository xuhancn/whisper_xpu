#include "audio_capture.h"
#include "portaudio.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

struct AudioCapture::Impl {
    PaStream* stream = nullptr;
    bool active = false;
    int sampleRate = 16000;
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

std::vector<std::string> AudioCapture::enumerate_devices() {
    std::vector<std::string> devices;

    // Initialize if not already done (static function may be called standalone)
    PaError err = Pa_Initialize();
    bool need_term = (err == paNoError);

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        if (need_term) Pa_Terminate();
        return devices;
    }

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxInputChannels > 0) {
            char buf[256];
            snprintf(buf, sizeof(buf), "[%d] %s (in: %d, sr: %.0f)",
                     i, info->name,
                     info->maxInputChannels,
                     info->defaultSampleRate);
            devices.push_back(buf);
        }
    }

    if (need_term) Pa_Terminate();
    return devices;
}

bool AudioCapture::start(int sample_rate, int frames_per_buffer) {
    if (pimpl_->active) return true;

    // Use the default input device
    int device = Pa_GetDefaultInputDevice();
    if (device == paNoDevice) {
        fprintf(stderr, "[audio] No default input device found\n");
        return false;
    }

    PaStreamParameters inputParams;
    memset(&inputParams, 0, sizeof(inputParams));
    inputParams.device = device;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(
        &pimpl_->stream,
        &inputParams,
        nullptr,                // no output
        sample_rate,
        frames_per_buffer,
        paClipOff,              // don't clip out-of-range samples
        Impl::paCallback,
        pimpl_.get()
    );

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
    pimpl_->active = true;
    fprintf(stderr, "[audio] Capture started (device %d, %d Hz)\n", device, sample_rate);
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

bool AudioCapture::is_active() const {
    return pimpl_->active;
}

int AudioCapture::sample_rate() const {
    return pimpl_->sampleRate;
}
