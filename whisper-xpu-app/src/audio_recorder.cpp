#include "src/audio_recorder.h"
#include "../audio_capture.h"
#include "engine.h"

#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
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
    join_thread(m_vadThread);
    join_thread(m_transcribeThread);
    wxLogMessage("[AudioRecorder] destroyed");
}

// ────────────────────────────────────────────────────────────
// Start / Stop
// ────────────────────────────────────────────────────────────

bool AudioRecorder::start(int micIndex) {
    if (m_recording) return true;

    m_stopping = false;
    m_vadDone = false;
    m_chunkQueue.clear();
    m_ring.clear();
    m_capture = std::make_unique<AudioCapture>();
    m_capture->set_callback([this](const float* s, size_t n) -> size_t {
        on_audio_cb(this, s, n);
        return n;
    });

    bool ok = m_capture->start(micIndex, 16000, 512);
    wxLogMessage("[AudioRecorder] start mic=%d ok=%d", micIndex, (int)ok);
    if (!ok) { m_capture.reset(); return false; }

    m_recording = true;
    m_vadThread = std::thread(&AudioRecorder::vad_loop, this);
    m_transcribeThread = std::thread(&AudioRecorder::transcribe_loop, this);
    return true;
}

void AudioRecorder::stop() {
    if (!m_recording) return;

    wxLogMessage("[AudioRecorder] stop");
    m_recording = false;
    m_stopping  = true;   // aborts in-flight whisper_full + makes both loops exit

    // Signal VAD is done so the transcribe consumer wakes even if the queue
    // is empty (it then drains remaining queued chunks before exiting).
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        m_vadDone = true;
    }
    m_queueCv.notify_all();

    // Join both threads before tearing down capture so neither touches `this`
    // / m_engine after destruction — OnToggleRecord calls m_recorder.reset()
    // the instant stop() returns.  The in-flight transcribe_chunk (if any)
    // aborts in milliseconds via the abort flag.
    join_thread(m_vadThread);
    join_thread(m_transcribeThread);

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }

    wxLogMessage("[AudioRecorder] stopped");
}

// ────────────────────────────────────────────────────────────
// VAD producer loop: ring → energy VAD → chunk queue (never blocks on transcription)
// ────────────────────────────────────────────────────────────

// Energy-based voice activity detection.  Model-free, millisecond-cost, and
// naturally skips silence (which caused 0-char output when arbitrary windows
// were fed to whisper).  A speech chunk is closed by a sustained silence gap
// or a max-duration cap; each closed chunk is queued for transcription.
//
// This loop must NEVER call transcribe_chunk — that takes ~4-8s on CPU and
// would stall the ring pull, overflowing the 10s ring and dropping live
// speech.  Transcription happens on the separate transcribe_loop thread.

void AudioRecorder::vad_loop() {
    wxLogMessage("[AudioRecorder] vad loop started");

    constexpr int   SR              = 16000;
    constexpr int   FRAME           = 320;                 // 20 ms
    constexpr float ENERGY_THRESH   = 0.0035f;             // RMS; ~tunable
    constexpr int   MIN_SILENCE     = 18;                  // 360 ms closes a chunk
    constexpr int   MAX_SPEECH      = SR * 8 / FRAME;      // 8 s hard cap
    constexpr size_t MIN_TRANSCRIBE = (size_t)(SR * 0.5);  // skip <0.5s blips

    std::vector<float> frame(FRAME);
    std::vector<float> chunk;
    chunk.reserve(SR * 2);

    bool inSpeech = false;
    int  silenceRun = 0;
    int  chunkFrames = 0;
    int  chunkNo = 0;
    float emin = 1e9f, emax = 0.f;

    // Per-second energy summary — independent of chunk firing, so we can see
    // actual mic levels and tune ENERGY_THRESH even when no speech fires.
    int   sumFrames = 0, sumVoiced = 0;
    float sumMin = 1e9f, sumMax = 0.f;

    while (m_recording) {
        size_t n = m_ring.pull(frame.data(), FRAME);
        if (n == 0) {
            std::this_thread::sleep_for(20ms);   // wait for audio to arrive
            continue;
        }
        for (size_t i = n; i < (size_t)FRAME; ++i) frame[i] = 0.f;

        // Frame RMS energy.
        float se = 0.f;
        for (int i = 0; i < FRAME; ++i) se += frame[i] * frame[i];
        float rms = std::sqrt(se / FRAME);
        bool voiced = (rms > ENERGY_THRESH);

        // Per-second energy summary.
        sumFrames++;
        if (voiced) sumVoiced++;
        if (rms < sumMin) sumMin = rms;
        if (rms > sumMax) sumMax = rms;
        if (sumFrames >= 50) {
            size_t queued;
            { std::lock_guard<std::mutex> lk(m_queueMutex); queued = m_chunkQueue.size(); }
            wxLogMessage("[AudioRecorder] energy 1s: rms min=%.5f max=%.5f voiced=%d/50 ring=%d queue=%d",
                         sumMin, sumMax, sumVoiced, (int)m_ring.available(), (int)queued);
            sumFrames = 0; sumVoiced = 0; sumMin = 1e9f; sumMax = 0.f;
        }

        if (!inSpeech) {
            if (voiced) {
                inSpeech = true;
                chunk.assign(frame.begin(), frame.end());
                chunkFrames = 1;
                silenceRun = 0;
                emin = rms; emax = rms;
            }
            continue;
        }

        // inSpeech — accumulate.
        chunk.insert(chunk.end(), frame.begin(), frame.end());
        chunkFrames++;
        if (rms < emin) emin = rms;
        if (rms > emax) emax = rms;
        if (voiced) silenceRun = 0; else silenceRun++;

        bool flush = (silenceRun >= MIN_SILENCE) || (chunkFrames >= MAX_SPEECH);
        if (!flush) continue;

        // Close chunk.  Skip transcription for blips.
        chunkNo++;
        double dur = (double)chunk.size() / SR;
        wxLogMessage("[AudioRecorder] chunk %d queued: %.2fs (emin=%.5f emax=%.5f)",
                     chunkNo, dur, emin, emax);

        if (chunk.size() >= MIN_TRANSCRIBE) {
            std::lock_guard<std::mutex> lk(m_queueMutex);
            m_chunkQueue.push_back(std::move(chunk));
            m_queueCv.notify_one();
        }

        inSpeech = false;
        chunk.clear();
        silenceRun = 0;
        chunkFrames = 0;
        emin = 1e9f; emax = 0.f;
    }

    // Mark VAD done so the transcribe consumer can drain + exit.
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        m_vadDone = true;
    }
    m_queueCv.notify_all();
    wxLogMessage("[AudioRecorder] vad loop exiting");
}

// ────────────────────────────────────────────────────────────
// Transcribe consumer loop: chunk queue → transcribe_chunk → on_text
// ────────────────────────────────────────────────────────────

void AudioRecorder::transcribe_loop() {
    wxLogMessage("[AudioRecorder] transcribe loop started");
    int chunkNo = 0;

    while (true) {
        std::vector<float> chunk;
        {
            std::unique_lock<std::mutex> lk(m_queueMutex);
            m_queueCv.wait(lk, [this] {
                return m_stopping.load() || m_vadDone || !m_chunkQueue.empty();
            });
            if (m_chunkQueue.empty()) {
                // Awake due to stop or VAD-done with empty queue → exit.
                if (m_stopping.load() || m_vadDone) break;
                continue;
            }
            chunk = std::move(m_chunkQueue.front());
            m_chunkQueue.erase(m_chunkQueue.begin());
        }

        if (m_stopping.load()) {
            // Drain remaining queued chunks quickly without transcribing on
            // abort — or transcribe a final chunk if it has real content?
            // On stop we abort: skip pending chunks (user wants fast exit).
            break;
        }

        chunkNo++;
        double dur = (double)chunk.size() / 16000.0;
        wxLogMessage("[AudioRecorder] transcribe chunk %d: %.2fs ...", chunkNo, dur);

        std::string text = m_engine->transcribe_chunk(chunk.data(),
                                                      (int)chunk.size(),
                                                      &m_stopping);
        wxLogMessage("[AudioRecorder] transcribe chunk %d: %d chars", chunkNo, (int)text.size());
        if (!text.empty() && m_on_text) m_on_text(text);
    }

    wxLogMessage("[AudioRecorder] transcribe loop exiting");
}
