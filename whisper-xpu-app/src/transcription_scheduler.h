#pragma once

#include "src/ring_buffer.h"
#include "engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class AudioCapture;

// Delivers finalized, in-order, overlap-deduped transcript text for one 5s
// chunk.  Called from the merger thread ⇒ must be thread-safe (the app posts
// to the UI via wxWindow::CallAfter).
using TextCallback = std::function<void(const std::string&)>;

// Per-chunk emission hook for tests/instrumentation: chunk index (in-order,
// 0-based) + finalized text.  Optional; default unset (no effect on the app).
using ChunkCallback = std::function<void(int, const std::string&)>;

// Pipelined replacement for AudioRecorder.  Transcribes ~7s (CPU turbo)
// windows in parallel with capture so continuous speech transcribes
// near-real-time, instead of serially queueing behind one slow
// transcribe_chunk.
//
//   AudioCapture (PortAudio thread) → RingBuffer<float>
//   Windower  → 6s windows (1s overlap tail + 5s new), one every 5s
//               → worker queue, indexed, with a global pcm_start time
//   Pool (4)  → shares ONE Engine (one whisper_context = one read-only model
//               copy); each worker owns its own whisper_state (KV cache +
//               compute buffers) and runs whisper_full_with_state.  This is
//               whisper.cpp's "one context + N states" pattern (issue #523):
//               pool RAM stays ~1× model + 4 small states instead of 4× model.
//               n_threads per worker = cores/4 (4×5=20).  Pops a window,
//               Engine::transcribe_window_with_state(&m_stopping), posts → merger.
//   Merger    → ordered map; emits chunk K only once K-1 emitted (in-order UI
//               text despite out-of-order completion).  Overlap dedup: keep a
//               segment iff its global midpoint ∈ [K*5000+250, K*5000+4750]ms
//               — 0.25s guards at each 5s boundary ⇒ no duplicate words and
//               no gaps (boundary-word splits are a documented rare edge case).
//   stop()    → m_stopping aborts in-flight transcribe_window_with_state via
//               the flag; joins all 6 threads, frees the 4 states.  The
//               shared Engine is BORROWED from the app and is NOT released
//               here — one model copy for the whole recording session.
//
// Two launch modes:
//   start(micIndex, sharedEngine, device)  — app path: PortAudio captures into
//      the ring; borrows the app's m_engine as the shared context.
//   start_no_capture(sharedEngine, device) — headless path: caller feeds PCM
//      via feed_audio() (e.g. a unit test replaying a WAV through the EXACT
//      same windower/pool/merger, no mic, no UI).  Pacing is the caller's
//      concern.  `sharedEngine` is borrowed, not owned.
//
// Window→global time: every window is 6s with a 1s prefix (the previous
// window's last 1s, or silence for window 0), so pcm_start_global_ms =
// index*5000 - 1000.  Segment midpoints (chunk-local, from transcribe_window)
// + pcm_start_global_ms ⇒ global midpoint ⇒ the dedup band above.
class TranscriptionScheduler {
public:
    // Readable pipeline counters for test asserts / status display.
    struct PipelineStats {
        int  windows_dispatched = 0;   // windower pushed onto the worker queue
        int  windows_completed  = 0;   // workers finished + posted to merger
        int  workers_used_mask  = 0;   // bit w set ⇒ worker w ran ≥1 window
        int  segments_kept      = 0;   // merger kept (midpoint in core band)
        int  segments_dropped   = 0;   // merger dropped (overlap guard band)
        long total_chars        = 0;   // chars emitted via on_text
        int  next_emit          = 0;   // next in-order chunk index to emit
        int  pending            = 0;   // results held awaiting in-order drain
        int  last_stop_ms       = 0;   // wall time of the last stop() call
    };

    // ring_capacity: samples the ring holds (default 10s @ 16 kHz).  A
    // file-replay test passes the whole clip length so it can pre-load it.
    TranscriptionScheduler(TextCallback on_text, size_t ring_capacity = 10 * 16000);
    ~TranscriptionScheduler();

    TranscriptionScheduler(const TranscriptionScheduler&) = delete;
    TranscriptionScheduler& operator=(const TranscriptionScheduler&) = delete;

    // App path: opens the microphone via PortAudio and starts all threads.
    // `sharedEngine` is borrowed (e.g. the app's m_engine) — one read-only
    // model copy for the whole session; not released on stop().  The engine
    // already carries its device (CPU/GPU), so no device arg here.
    bool start(int micIndex, whisper_xpu::Engine& sharedEngine);

    // Headless path: same threads, no AudioCapture.  Caller pushes PCM via
    // feed_audio() (e.g. from a WAV).  Returns false on state-creation
    // failure.  `sharedEngine` is borrowed, not owned.
    bool start_no_capture(whisper_xpu::Engine& sharedEngine);

    // Push 16 kHz mono PCM into the ring — same entry point the PortAudio
    // callback uses.  For the headless/file-replay path.
    void feed_audio(const float* samples, size_t count);

    void stop();
    bool is_recording() const { return m_recording.load(); }

    void set_on_chunk(ChunkCallback cb) { m_on_chunk = std::move(cb); }
    PipelineStats stats() const;

private:
    // Window handed windower → workers.
    struct WindowJob {
        int             index;               // chunk index (0-based)
        int64_t         pcm_start_global_ms; // global time of pcm[0]
        std::vector<float> pcm;              // 6s window (1s tail + 5s new)
    };
    // Result handed workers → merger (ChunkResult comes from engine.h).
    struct ChunkDone {
        int                     index;
        int64_t                 pcm_start_global_ms;
        whisper_xpu::ChunkResult result;
    };

    void windower_loop();
    void worker_loop(int workerId);
    void merger_loop();
    static void join_thread(std::thread& t);

    // Create the 4 per-worker whisper_state*s on the borrowed shared engine.
    // Returns false (clearing m_states) if any whisper_init_state fails.
    bool init_states();

    TextCallback m_on_text;
    ChunkCallback m_on_chunk;

    // Worker pool — ONE borrowed Engine (shared read-only model context) + 4
    // per-worker whisper_state*s (own KV cache + compute buffers), created in
    // start()/start_no_capture(), freed in stop().  The Engine itself is
    // borrowed (e.g. the app's m_engine) and NOT released here.
    static constexpr int POOL_SIZE = 4;
    whisper_xpu::Engine*            m_sharedEngine = nullptr;  // non-owning
    std::vector<whisper_state*>     m_states;                   // POOL_SIZE, owned
    std::vector<std::string>        m_workerLang;               // POOL_SIZE, per-worker lang cache
    int m_nThreadsPerWorker = 4;

    // Capture + ring (PortAudio thread → ring → windower).  In headless mode
    // m_capture stays null and the caller feeds the ring via feed_audio().
    RingBuffer<float>               m_ring;
    std::unique_ptr<AudioCapture>   m_capture;
    static void on_audio_cb(TranscriptionScheduler* self,
                            const float* samples, size_t count);

    // Windower → workers queue.
    std::deque<WindowJob>       m_workerQueue;
    std::mutex                  m_workerMutex;
    std::condition_variable     m_workerCv;

    // Workers → merger queue.
    std::deque<ChunkDone>       m_mergerQueue;
    mutable std::mutex          m_mergerMutex;   // mutable: stats() const locks it
    std::condition_variable     m_mergerCv;

    // Merger ordered state.
    std::map<int, ChunkDone>    m_pending;   // index → result (awaiting in-order drain)
    int                         m_nextEmit = 0;

    std::thread                 m_windowerThread;
    std::vector<std::thread>    m_workerThreads;
    std::thread                 m_mergerThread;

    std::atomic<bool>           m_recording{false};
    std::atomic<bool>           m_stopping{false};  // aborts in-flight whisper_full + exits all loops

    // Introspection counters (populated across threads; read via stats()).
    std::atomic<int>  m_dispatched{0};
    std::atomic<int>  m_completed{0};
    std::atomic<int>  m_workersUsed{0};   // bitmask
    std::atomic<int>  m_segsKept{0};
    std::atomic<int>  m_segsDropped{0};
    std::atomic<long> m_charsEmitted{0};
    std::atomic<int>  m_lastStopMs{0};
};
