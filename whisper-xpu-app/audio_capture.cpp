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

// Linear resampling state.  Only used when the device cannot be opened at
// the requested (16 kHz) rate and we fall back to its native rate.
struct ResampleState {
    bool   active = false;       // true ⇒ resample inputRate → outputRate
    int    inputRate = 16000;    // actual PortAudio stream rate
    int    outputRate = 16000;   // rate delivered to the callback (always 16k)
    double pos = 0.0;            // fractional input index carried across callbacks [0,1)
    std::vector<float> scratch;  // resampled output buffer (callback thread only)
};

struct AudioCapture::Impl {
    PaStream* stream = nullptr;
    bool active = false;
    int sampleRate = 16000;   // logical output rate reported to callers
    int deviceId = kMicDefault;
    AudioCaptureCallback callback;
    ResampleState resample;

    static int paCallback(const void* input, void* output,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void* userData) {
        auto* impl = static_cast<Impl*>(userData);
        if (!impl || !impl->callback || !input) return paContinue;

        const float* in = static_cast<const float*>(input);

        if (!impl->resample.active) {
            // Passthrough — device already runs at the requested rate.
            impl->callback(in, frameCount);
        } else {
            // Resample inputRate → outputRate (16 kHz) via linear
            // interpolation, carrying the fractional position across
            // callbacks so the seam between blocks is continuous.
            const double step = (double)impl->resample.inputRate
                               / (double)impl->resample.outputRate; // in-samples per out-sample
            size_t maxOut = (size_t)((double)frameCount / step + 16.0);
            if (impl->resample.scratch.size() < maxOut)
                impl->resample.scratch.resize(maxOut);

            size_t outCount = 0;
            size_t inIdx = 0;
            double pos = impl->resample.pos;
            while (inIdx + 1 < frameCount) {
                float s = (float)(in[inIdx] + (in[inIdx + 1] - in[inIdx]) * pos);
                if (outCount >= impl->resample.scratch.size()) break;
                impl->resample.scratch[outCount++] = s;
                pos += step;
                while (pos >= 1.0) { pos -= 1.0; ++inIdx; }
            }
            impl->resample.pos = pos;   // carry [0,1) into next callback

            if (outCount > 0)
                impl->callback(impl->resample.scratch.data(), outCount);
        }

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

// Some host APIs (notably WASAPI) expose render-endpoint "loopback" devices
// as if they were inputs.  They capture what the speakers are playing, not
// the microphone, and their capture clock stalls when nothing is playing —
// so a recording opened on one never receives callbacks (available stays 0).
// Exclude them from the microphone list.
static bool is_loopback_device(const char* name) {
    if (!name) return false;
    std::string s(name);
    // "[Loopback]" is the suffix PortAudio's WASAPI backend appends.
    return s.find("[Loopback]") != std::string::npos
        || s.find("[loopback]") != std::string::npos;
}

static bool is_loopback_device(int device_index) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(device_index);
    return info ? is_loopback_device(info->name) : false;
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

        // Skip render-endpoint loopback devices — see is_loopback_device().
        if (is_loopback_device(info->name))
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
        // A loopback can be the PortAudio "default input" on systems with
        // no real mic set as default — don't propagate that flag to it.
        dev.is_default   = (i == default_dev && !is_loopback_device(default_dev));
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
        // The system default can be a WASAPI loopback (render endpoint).
        // Those never deliver samples when nothing is playing, so fall
        // back to the first real input device instead.
        if (device == paNoDevice || is_loopback_device(device)) {
            int n = Pa_GetDeviceCount();
            device = paNoDevice;
            for (int i = 0; i < n; ++i) {
                const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
                if (!info || info->maxInputChannels < 1) continue;
                if (is_loopback_device(info->name)) continue;
                device = i;
                break;
            }
            if (device == paNoDevice) {
                fprintf(stderr, "[audio] No real input device found (only loopbacks?)\n");
                return false;
            }
            fprintf(stderr, "[audio] Default was loopback/none; using device %d\n", device);
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

    // The callback always delivers `sample_rate` (16 kHz) to the caller.
    // If the device refuses 16 kHz (common on WASAPI shared mode), open at
    // its native default rate and resample to 16 kHz in the callback.
    pimpl_->resample.outputRate = sample_rate;
    pimpl_->resample.active = false;
    pimpl_->resample.pos = 0.0;

    int inputRate = sample_rate;

    PaError err = Pa_OpenStream(
        &pimpl_->stream, &inputParams, nullptr,
        sample_rate, frames_per_buffer, paClipOff,
        Impl::paCallback, pimpl_.get());

    if (err != paNoError) {
        int nativeRate = (int)devInfo->defaultSampleRate;
        if (nativeRate <= 0 || nativeRate == sample_rate) {
            fprintf(stderr, "[audio] OpenStream error at %d Hz: %s\n",
                    sample_rate, Pa_GetErrorText(err));
            return false;
        }
        fprintf(stderr, "[audio] %d Hz unsupported (0x%x); retrying native %d Hz + resample\n",
                sample_rate, (unsigned)err, nativeRate);
        err = Pa_OpenStream(
            &pimpl_->stream, &inputParams, nullptr,
            nativeRate, frames_per_buffer, paClipOff,
            Impl::paCallback, pimpl_.get());
        if (err != paNoError) {
            fprintf(stderr, "[audio] OpenStream error at native %d Hz: %s\n",
                    nativeRate, Pa_GetErrorText(err));
            return false;
        }
        inputRate = nativeRate;
        pimpl_->resample.inputRate = nativeRate;
        pimpl_->resample.active = true;
    }

    err = Pa_StartStream(pimpl_->stream);
    if (err != paNoError) {
        fprintf(stderr, "[audio] StartStream error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(pimpl_->stream);
        pimpl_->stream = nullptr;
        return false;
    }

    pimpl_->sampleRate = sample_rate;   // logical output rate (what callers get)
    pimpl_->deviceId   = device;
    pimpl_->active     = true;
    fprintf(stderr, "[audio] Capture started (device %d: %s, %d Hz%s)\n",
            device, devInfo->name, inputRate,
            pimpl_->resample.active ? "  >resample> 16000" : "");
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
