#include "src/audio_recorder.h"
#include "../audio_capture.h"
#include "engine.h"

#include <chrono>
#include <cstring>
#include <wx/log.h>

using namespace std::chrono_literals;

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────

void AudioRecorder::join_thread(std::thread& t) {
    if (t.joinable()) t.join();
}

// Static PortAudio callback → pushes to ring buffer
void AudioRecorder::on_audio_cb(AudioRecorder* self, const float* samples, size_t count) {
    self->m_ring.push(samples, count);
}

// ────────────────────────────────────────────────────────────
// Constructor / Destructor
// ────────────────────────────────────────────────────────────

AudioRecorder::AudioRecorder(whisper_xpu::Engine* engine, TextCallback on_text)
    : m_engine(engine)
    , m_on_text(std::move(on_text))
    , m_ring(10 * 16000)  // 10s at 16kHz
{
    wxLogMessage("[AudioRecorder] created");
}

AudioRecorder::~AudioRecorder() {
    if (m_recording) stop();
    join_thread(m_pollThread);
    join_thread(m_transcribeThread);
    wxLogMessage("[AudioRecorder] destroyed");
}

// ────────────────────────────────────────────────────────────
// Start / Stop
// ────────────────────────────────────────────────────────────

void AudioRecorder::start(int micIndex) {
    if (m_recording) return;

    m_ring.clear();
    m_capture = std::make_unique<AudioCapture>();
    m_capture->set_callback([this](const float* s, size_t n) -> size_t {
        on_audio_cb(this, s, n);
        return n;
    });

    bool ok = m_capture->start(micIndex, 16000, 512);
    wxLogMessage("[AudioRecorder] start mic=%d ok=%d", micIndex, (int)ok);
    if (!ok) { m_capture.reset(); return; }

    m_recording = true;
    m_pollThread = std::thread(&AudioRecorder::record_loop, this);
}

void AudioRecorder::stop() {
    if (!m_recording) return;

    wxLogMessage("[AudioRecorder] stop");
    m_recording = false;

    // Wake poll thread (it's in sleep_for)
    // join_thread will block until the poll loop exits naturally
    join_thread(m_pollThread);

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }

    // Wait for in-flight transcription, then final flush (async)
    while (m_transcribing.load())
        std::this_thread::sleep_for(50ms);

    join_thread(m_transcribeThread);
    transcribe_ring();

    // Wait for final flush
    while (m_transcribing.load())
        std::this_thread::sleep_for(50ms);
    join_thread(m_transcribeThread);

    wxLogMessage("[AudioRecorder] stopped");
}

// ────────────────────────────────────────────────────────────
// Polling loop (runs in m_pollThread)
// ────────────────────────────────────────────────────────────

void AudioRecorder::record_loop() {
    wxLogMessage("[AudioRecorder] poll loop started");
    while (m_recording) {
        std::this_thread::sleep_for(2s);
        if (!m_recording) break;
        transcribe_ring();
    }
    wxLogMessage("[AudioRecorder] poll loop exiting");
}

// ────────────────────────────────────────────────────────────
// Pull from ring → transcribe_stream → callback
// ────────────────────────────────────────────────────────────

void AudioRecorder::transcribe_ring() {
    // Don't stack overlapping transcription runs
    if (m_transcribing.exchange(true)) {
        wxLogMessage("[AudioRecorder] transcribe: already running, skip");
        return;
    }

    size_t avail = m_ring.available();
    wxLogMessage("[AudioRecorder] transcribe: available=%d samples (%.1fs)",
                 (int)avail, (double)avail / 16000.0);

    constexpr size_t kMinSamples = 8000;  // 0.5s
    if (avail < kMinSamples) {
        wxLogMessage("[AudioRecorder] transcribe: too few samples, skip");
        m_transcribing = false;
        return;
    }

    // Run transcription in background thread
    join_thread(m_transcribeThread);
    m_transcribeThread = std::thread([this]() {
        // Pull callback for Engine::transcribe_stream
        size_t pulled = 0;
        std::vector<float> chunk(48000);  // 3s buffer
        auto stream_cb = [this, &chunk, &pulled](float* buf, size_t max) -> size_t {
            auto n = m_ring.pull(buf, max);
            pulled += n;
            return n;
        };
        (void)chunk;

        wxLogMessage("[AudioRecorder] calling transcribe_stream...");
        auto result = m_engine->transcribe_stream(stream_cb);
        wxLogMessage("[AudioRecorder] transcribe_stream done: %d chars, %d samples, %.0fms",
                     (int)result.text.size(), (int)pulled, result.processing_time_ms);

        if (!result.text.empty() && m_on_text) {
            m_on_text(result.text);
        }

        m_transcribing = false;
    });
}
