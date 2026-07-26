#pragma once

#include "src/ring_buffer.h"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <string>

class AudioCapture;
namespace whisper_xpu { class Engine; }

// Callback: delivers transcribed text from the recording thread.
// Must be thread-safe (typically posts to UI via CallAfter).
using TextCallback = std::function<void(const std::string&)>;

// wxWidgets-free audio recorder.  Uses a polling thread (std::thread)
// instead of wxTimer so it can be controlled from anywhere — UI button,
// hotkey handler, or CLI — without wx coupling.
//
// Architecture:
//   AudioCapture (PortAudio thread) → RingBuffer<float> (10s, 16kHz)
//   Polling thread (every 2s) → Engine::transcribe_stream()
//   Transcription runs in a background thread → TextCallback
class AudioRecorder {
public:
    AudioRecorder(whisper_xpu::Engine* engine, TextCallback on_text);
    ~AudioRecorder();

    void start(int micIndex);
    void stop();
    bool is_recording() const { return m_recording.load(); }

private:
    void record_loop();           // runs in m_pollThread — polls ring every 2s
    void transcribe_ring();       // pulls ring → transcribe_stream → callback

    static void join_thread(std::thread& t);

    whisper_xpu::Engine*    m_engine;
    TextCallback            m_on_text;
    RingBuffer<float>       m_ring;
    std::unique_ptr<AudioCapture> m_capture;

    // Audio callback: PortAudio thread → ring buffer
    static void on_audio_cb(AudioRecorder* self, const float* samples, size_t count);

    std::thread       m_pollThread;        // 2s polling loop
    std::thread       m_transcribeThread;  // background transcription
    std::atomic<bool> m_recording{false};
    std::atomic<bool> m_transcribing{false};
};
