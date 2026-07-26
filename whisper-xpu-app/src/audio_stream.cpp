#include "src/audio_stream.h"
#include "../audio_capture.h"
#include "engine.h"

#include <wx/timer.h>
#include <wx/textctrl.h>
#include <wx/log.h>
#include <algorithm>
#include <cstring>

// Rate-limit OnAudioData logging: only log every Nth call
static constexpr int kAudioLogInterval = 100;

// ────────────────────────────────────────────────────────────
// Constructor / Destructor
// ────────────────────────────────────────────────────────────

AudioStream::AudioStream(whisper_xpu::Engine* engine, wxTextCtrl* output)
    : m_engine(engine)
    , m_textOutput(output)
    , m_ring(kRingSize, 0.0f)
{
    m_timer = new wxTimer(m_textOutput);  // owner = wxWindow for event loop
    m_timer->Bind(wxEVT_TIMER, &AudioStream::OnTimer, this);
    wxLogMessage("[AudioStream] created, engine=%p, output=%p", engine, output);
}

AudioStream::~AudioStream() {
    if (m_recording) stop();
    delete m_timer;
    wxLogMessage("[AudioStream] destroyed");
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

    bool ok = m_capture->start(micIndex, 16000, 512);
    wxLogMessage("[AudioStream] start called, mic=%d, result=%d", micIndex, (int)ok);

    if (!ok) {
        m_capture.reset();
        return;
    }

    m_recording = true;
    m_timer->Start(2000);
    wxLogMessage("[AudioStream] timer started, interval=2000ms");
}

void AudioStream::stop() {
    if (!m_recording) return;

    wxLogMessage("[AudioStream] stop called");
    m_timer->Stop();

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }

    // Flush remaining audio
    ProcessAccumulatedAudio();
    m_recording = false;
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
    // Rate-limited log
    static std::atomic<int> callCount{0};
    int n = ++callCount;
    if (n % kAudioLogInterval == 0)
        wxLogMessage("[AudioStream] OnAudioData: %d samples (call #%d)", (int)count, n);
}

// ────────────────────────────────────────────────────────────
// Timer (main thread, every 2s)
// ────────────────────────────────────────────────────────────

void AudioStream::OnTimer(wxTimerEvent& /*event*/) {
    if (!m_recording) return;
    // Count available samples for log
    size_t avail;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_writePos >= m_readPos)
            avail = m_writePos - m_readPos;
        else
            avail = kRingSize - m_readPos + m_writePos;
    }
    wxLogMessage("[AudioStream] timer fired, available=%d samples", (int)avail);
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

    wxLogMessage("[AudioStream] ProcessAccumulatedAudio: available=%d samples (%.1fs)",
               (int)available, (double)available / 16000.0);

    // Minimum ~0.5 s of audio to be worth transcribing
    constexpr size_t kMinSamples = 8000;  // 0.5s @ 16kHz
    if (available < kMinSamples) {
        wxLogMessage("[AudioStream] skipping — only %d samples (< %d min)", (int)available, (int)kMinSamples);
        return;
    }

    // Provide audio to Engine via pull callback
    size_t pulled = 0;
    auto stream_cb = [this, &pulled](float* buffer, size_t max_samples) -> size_t {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t n = 0;
        while (n < max_samples && m_readPos != m_writePos) {
            buffer[n++] = m_ring[m_readPos];
            m_readPos = (m_readPos + 1) % kRingSize;
        }
        pulled += n;
        return n;
    };

    wxLogMessage("[AudioStream] calling transcribe_stream...");
    whisper_xpu::TranscriptionResult result = m_engine->transcribe_stream(stream_cb);
    wxLogMessage("[AudioStream] transcribe_stream returned: %d chars, pulled=%d samples, time=%.0fms",
               (int)result.text.size(), (int)pulled, result.processing_time_ms);

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
