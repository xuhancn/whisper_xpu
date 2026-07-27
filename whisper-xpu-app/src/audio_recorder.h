#pragma once

#include "src/ring_buffer.h"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>

class AudioCapture;
namespace whisper_xpu { class Engine; }

// Callback: delivers transcribed text from the recording thread.
// Must be thread-safe (typically posts to UI via CallAfter).
using TextCallback = std::function<void(const std::string&)>;

// wxWidgets-free audio recorder.  Uses two worker threads (std::thread)
// instead of wxTimer so it can be controlled from anywhere — UI button,
// hotkey handler, or CLI — without wx coupling.
//
// Architecture (producer/consumer — keeps the ring drained even while a
// slow CPU transcription is in flight, so no speech is dropped):
//   AudioCapture (PortAudio thread) → RingBuffer<float> (10s, 16kHz)
//   VAD thread:   drains ring frame-by-frame → energy VAD → pushes closed
//                 speech chunks onto a thread-safe queue.  NEVER blocks on
//                 transcription, so the ring can't overflow while
//                 transcribe_chunk runs on CPU (which takes ~4-8s per chunk).
//   Transcribe thread: pops chunks from the queue → Engine::transcribe_chunk()
//                 → TextCallback, one at a time.
//   stop() aborts the in-flight transcribe_chunk (via the abort flag) and
//   joins both threads before returning — no use-after-free.
class AudioRecorder {
public:
    AudioRecorder(whisper_xpu::Engine* engine, TextCallback on_text);
    ~AudioRecorder();

    bool start(int micIndex);
    void stop();
    bool is_recording() const { return m_recording.load(); }

private:
    void vad_loop();          // producer: ring → VAD → chunk queue
    void transcribe_loop();   // consumer: queue → transcribe_chunk → on_text

    static void join_thread(std::thread& t);

    // Thread-safe queue of speech chunks waiting to be transcribed.
    std::vector<std::vector<float>> m_chunkQueue;
    std::mutex                      m_queueMutex;
    std::condition_variable         m_queueCv;
    bool                            m_vadDone = false;   // signals no more chunks coming

    whisper_xpu::Engine*    m_engine;
    TextCallback            m_on_text;
    RingBuffer<float>       m_ring;
    std::unique_ptr<AudioCapture> m_capture;

    // Audio callback: PortAudio thread → ring buffer
    static void on_audio_cb(AudioRecorder* self, const float* samples, size_t count);

    std::thread       m_vadThread;        // producer: ring → chunks
    std::thread       m_transcribeThread; // consumer: chunks → text
    std::atomic<bool> m_recording{false};
    std::atomic<bool> m_stopping{false};  // set by stop() to abort in-flight whisper_full + exit both threads
};
