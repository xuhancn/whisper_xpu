#pragma once

#include <vector>
#include <memory>
#include <mutex>

class wxTextCtrl;
class wxTimer;
class wxTimerEvent;
class AudioCapture;

namespace whisper_xpu {
    class Engine;
}

// Real-time audio recording with periodic transcription.
//
// PortAudio callback pushes float32 16kHz mono samples into a 10-second
// ring buffer.  A wxTimer fires every 2 seconds on the main thread,
// pulls accumulated audio, and feeds it to Engine::transcribe_stream().
// Transcribed text is appended to the wxTextCtrl.
class AudioStream {
public:
    AudioStream(whisper_xpu::Engine* engine, wxTextCtrl* output);
    ~AudioStream();

    void start(int micIndex);
    void stop();
    bool is_recording() const { return m_recording; }

private:
    void OnAudioData(const float* samples, size_t count);  // PortAudio thread
    void OnTimer(wxTimerEvent& event);                     // main thread
    void ProcessAccumulatedAudio();                        // transcribe ring content

    // Ring buffer: 10 seconds at 16 kHz mono f32
    static constexpr size_t kRingSize = 10 * 16000;

    whisper_xpu::Engine*        m_engine;
    wxTextCtrl*                 m_textOutput;
    std::unique_ptr<AudioCapture> m_capture;
    wxTimer*                    m_timer = nullptr;

    // Ring buffer state — guarded by m_mutex
    std::mutex      m_mutex;
    std::vector<float> m_ring;
    size_t          m_writePos = 0;
    size_t          m_readPos  = 0;

    bool m_recording = false;
};
