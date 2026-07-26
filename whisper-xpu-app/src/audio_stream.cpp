#include "src/audio_stream.h"
#include "../audio_capture.h"
#include "engine.h"

#include <wx/timer.h>
#include <wx/textctrl.h>
#include <algorithm>
#include <cstring>

// ────────────────────────────────────────────────────────────
// Constructor / Destructor
// ────────────────────────────────────────────────────────────

AudioStream::AudioStream(whisper_xpu::Engine* engine, wxTextCtrl* output)
    : m_engine(engine)
    , m_textOutput(output)
    , m_ring(kRingSize, 0.0f)
{
    m_timer = new wxTimer();
    m_timer->Bind(wxEVT_TIMER, &AudioStream::OnTimer, this);
}

AudioStream::~AudioStream() {
    if (m_recording) stop();
    delete m_timer;
}

// ────────────────────────────────────────────────────────────
// Start / Stop
// ────────────────────────────────────────────────────────────

void AudioStream::start(int micIndex) {
    if (m_recording) return;

    // Reset ring buffer
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::fill(m_ring.begin(), m_ring.end(), 0.0f);
        m_writePos = 0;
        m_readPos  = 0;
    }

    m_capture = std::make_unique<AudioCapture>();
    m_capture->set_callback([this](const float* samples, size_t count) -> size_t {
        OnAudioData(samples, count);
        return count;
    });

    if (!m_capture->start(micIndex, 16000, 512)) {
        m_capture.reset();
        return;
    }

    m_recording = true;
    m_timer->Start(2000);
}

void AudioStream::stop() {
    if (!m_recording) return;

    m_timer->Stop();
    m_recording = false;

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }

    // Flush remaining audio
    ProcessAccumulatedAudio();
}

// ────────────────────────────────────────────────────────────
// PortAudio callback (audio thread)
// ────────────────────────────────────────────────────────────

void AudioStream::OnAudioData(const float* samples, size_t count) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (size_t i = 0; i < count; ++i) {
        m_ring[m_writePos] = samples[i];
        m_writePos = (m_writePos + 1) % kRingSize;
    }
}

// ────────────────────────────────────────────────────────────
// Timer (main thread, every 2s)
// ────────────────────────────────────────────────────────────

void AudioStream::OnTimer(wxTimerEvent& /*event*/) {
    if (!m_recording) return;
    ProcessAccumulatedAudio();
}

// ────────────────────────────────────────────────────────────
// Pull accumulated audio → transcribe → append to wxTextCtrl
// ────────────────────────────────────────────────────────────

void AudioStream::ProcessAccumulatedAudio() {
    // Determine how much unread audio is in the ring buffer
    size_t available;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_writePos >= m_readPos) {
            available = m_writePos - m_readPos;
        } else {
            available = kRingSize - m_readPos + m_writePos;
        }
    }

    // Minimum ~0.5 s of audio to be worth transcribing
    constexpr size_t kMinSamples = 8000;  // 0.5s @ 16kHz
    if (available < kMinSamples) return;

    // Provide audio to Engine via pull callback
    auto stream_cb = [this](float* buffer, size_t max_samples) -> size_t {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t n = 0;
        while (n < max_samples && m_readPos != m_writePos) {
            buffer[n++] = m_ring[m_readPos];
            m_readPos = (m_readPos + 1) % kRingSize;
        }
        return n;
    };

    whisper_xpu::TranscriptionResult result = m_engine->transcribe_stream(stream_cb);

    if (!result.text.empty()) {
        // Append to text control from main thread
        wxString text(result.text);
        // Trim trailing whitespace + add a space for readability
        text.Trim();
        if (!text.IsEmpty()) {
            m_textOutput->WriteText(text + wxT(" "));
        }
    }
}
