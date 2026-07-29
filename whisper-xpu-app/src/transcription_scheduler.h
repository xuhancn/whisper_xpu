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
// chunk.  Called from the merger thread ⇒ the app's callback must be
// thread-safe (e.g. marshal to the UI thread).  This is pure data delivery —
// the scheduler itself contains no UI / no wx / no CallAfter.
using TextCallback = std::function<void(const std::string&)>;

// Per-chunk emission hook for tests/instrumentation: chunk index (in-order,
// 0-based) + finalized text.  Optional; default unset (no effect on the app).
using ChunkCallback = std::function<void(int, const std::string&)>;

// The high-level state of the scheduler, polled by the UI's sync thread.
// Pure data — no wx, no threads.  The UI owns the polling; the scheduler just
// reports.  Compared with operator!= (cheap field-wise) so the UI can skip the
// refresh on no-op polls.
enum class SchedulerState {
    Idle,        // no model loaded, nothing happening
    Loading,     // bg load_loop building the Engine + warming up (startup/reload)
    Ready,       // Engine ready, not recording; Record is instant from here
    Recording,   // capture + pipeline threads running
    Failed,      // last load/setup failed (e.g. bad model path / SYCL error)
};
struct SchedulerStatus {
    SchedulerState state         = SchedulerState::Idle;
    std::string    model_name;     // basename of the model path, or ""
    std::string    device_desc;    // engine device_description(), or ""
    bool           gpu_ready       = false;  // Engine built + warmup done
    bool           recording       = false;
    long           total_chars     = 0;      // chars emitted via on_text

    bool operator==(const SchedulerStatus& o) const {
        return state == o.state && model_name == o.model_name &&
               device_desc == o.device_desc && gpu_ready == o.gpu_ready &&
               recording == o.recording && total_chars == o.total_chars;
    }
    bool operator!=(const SchedulerStatus& o) const { return !(*this == o); }
};

// PURE-API streaming transcription engine controller.  Owns the Engine,
// model load, GPU warmup, PortAudio capture, the 4-worker pool, and the
// merger — but exposes ONLY data + commands.  NO UI knowledge, NO wx
// includes, NO CallAfter, NO polling threads.  The frontend (wxWidgets GUI,
// CLI, hotkey daemon, …) owns the status-polling thread and renders
// SchedulerStatus; it sends commands (async_setup/reload/start/stop).
//
// Lifecycle (WeChat-style non-blocking):
//   async_setup(model,device) / ctor — starts a background std::thread
//                  (load_loop) that runs the Engine ctor (800MB model load,
//                  file I/O + ggml alloc) + init_states + warmup_states
//                  (SYCL first-kernel JIT, per-state buffers).  Returns
//                  immediately; never blocks the caller.  Reused for every
//                  later Record; reload() tears it down + restarts it.
//   start(mic)    — opens PortAudio into the ring immediately (<100ms,
//                  model-independent) and sets m_captureActive.  If the engine
//                  is already ready, launches the windower/worker/merger
//                  threads now; otherwise defers until load_loop() finishes
//                  warmup and calls launch_threads_if_ready().  Record is
//                  instant even if pressed during the startup load.
//   stop()        — m_stopping aborts in-flight transcribe_window_with_state,
//                  joins the pipeline threads, frees the 4 states.  The owned
//                  Engine survives stop() (one model copy per session).
//   reload(m,d)   — stop() if recording, abort+join the load thread, drop the
//                  old Engine + states, then async_setup() with the new model/
//                  device.  Immediate; UI stays responsive (same as startup).
//   dtor          — abort+join the load thread + pipeline threads, free engine.
//
//   AudioCapture (PortAudio thread) → RingBuffer<float> (20s @ 16kHz)
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
//
// Two launch modes:
//   start(micIndex)            — app path: PortAudio captures into the ring;
//      the scheduler owns its Engine (built on the bg load thread).  Deferred
//      pipeline launch if the engine isn't ready yet.
//   start_no_capture(engine)  — headless/test path: caller feeds PCM via
//      feed_audio() (e.g. a unit test replaying a WAV).  The caller constructs
//      the Engine and BORROWS it (the scheduler does NOT own it in this path,
//      so no bg load thread).  Pacing is the caller's concern.
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

    // App path: the scheduler OWNS its Engine.  model_path + device_index are
    // used to construct the Engine on the background load thread (started from
    // the ctor via async_setup).  ring_capacity defaults to 20s @ 16 kHz so
    // audio captured during the startup load survives until the pipeline
    // launches.  on_text is the merger→UI data delivery callback.
    TranscriptionScheduler(TextCallback on_text,
                            std::string model_path,
                            int device_index,
                            size_t ring_capacity = 20 * 16000);
    // Test path: the scheduler does NOT own an Engine (the test borrows one
    // via start_no_capture).  No background load thread is started.  The
    // caller passes only the ring capacity (e.g. a WAV's whole length).
    TranscriptionScheduler(TextCallback on_text, size_t ring_capacity);
    ~TranscriptionScheduler();

    TranscriptionScheduler(const TranscriptionScheduler&) = delete;
    TranscriptionScheduler& operator=(const TranscriptionScheduler&) = delete;

    // ── Commands (pure — return immediately, never block) ──

    // Start the background Engine ctor + warmup (also run from the owner ctor).
    // Idempotent if already loading/ready.  Returns false only if model_path is
    // empty.  Safe to call again after reload() tears the old setup down.
    bool async_setup(const std::string& model_path, int device_index);

    // Immediate reload: stop any recording, abort+join the load thread, drop
    // the old Engine + states, then async_setup() with the new model/device.
    // The UI's sync thread observes Loading → Ready/Failed.  Never blocks.
    void reload(const std::string& model_path, int device_index);

    // Open PortAudio (fast, model-independent) + set m_captureActive.  Launches
    // the pipeline threads now if the engine is ready, else defers until
    // load_loop() finishes.  Returns false only on mic-open failure.
    bool start(int micIndex);

    // Headless/test path: no AudioCapture, caller feeds PCM via feed_audio().
    // `sharedEngine` is BORROWED.  Returns false on state-creation failure.
    bool start_no_capture(whisper_xpu::Engine& sharedEngine);

    // Push 16 kHz mono PCM into the ring (headless/file-replay path).
    void feed_audio(const float* samples, size_t count);

    // Stop the current recording: abort in-flight work, join the pipeline
    // threads, free the 4 per-worker states.  The owned Engine survives.
    void stop();

    // ── Queries (pure data — the UI's sync thread polls this) ──

    bool is_recording() const { return m_recording.load(); }
    bool is_ready() const { return m_engineReady.load(); }
    // Snapshot of state + model/device names + gpu_ready/recording/total_chars.
    SchedulerStatus query_status() const;

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
    void load_loop();                          // background: Engine ctor + warmup
    static void join_thread(std::thread& t);

    // Active engine: the owned m_engine (app path) or the borrowed
    // m_sharedEngine (test path).  Exactly one is set at a time.  Returns
    // nullptr if neither is ready yet.
    whisper_xpu::Engine* engine() const;

    // Create the 4 per-worker whisper_state*s on the active engine.  Returns
    // false (clearing m_states) if any whisper_init_state fails.
    bool init_states();

    // Prime every per-worker state with one short silence window BEFORE the
    // pipeline threads start.  Forces SYCL first-kernel JIT + per-state buffer
    // alloc + (first-window) encoder pass on the background load thread, while
    // no real audio is queued.  Uses a THROWAWAY local `lang` (NOT
    // m_workerLang[i]) so whisper's silence language-detect default (typically
    // "en") is never cached — the real first window still auto-detects on real
    // speech.  `abort_flag` lets reload() interrupt the warmup mid-pass.
    void warmup_states(const std::atomic<bool>* abort_flag);

    // Launch the windower/worker/merger threads once BOTH the engine is ready
    // AND capture is active (or, for start_no_capture, the engine is set).
    // Idempotent + mutex-guarded.  Called from start() and from load_loop().
    void launch_threads_if_ready();

    // True in the owner path (ctor starts load_loop on m_loadThread).  The
    // test path sets this false so start_no_capture skips the load thread.
    bool m_ownsEngine = false;
    std::string m_modelPath;
    int         m_deviceIndex = 0;
    // Basename of m_modelPath, computed in async_setup (wx thread) and read by
    // query_status (sync thread) under m_engineMutex so the std::string read
    // can't tear.  query_status never touches m_modelPath directly.
    std::string m_modelName;

    TextCallback m_on_text;
    ChunkCallback m_on_chunk;

    // Worker pool — ONE Engine (owned in the app path, borrowed in the test
    // path) + 4 per-worker whisper_state*s (own KV cache + compute buffers),
    // created in start()/start_no_capture()/load_loop(), freed in stop().
    static constexpr int POOL_SIZE = 4;
    // m_engineMutex guards m_engine / m_sharedEngine against reload()/dtor
    // mutating them while worker/query_status threads read engine().  Workers
    // run only while m_engineReady (reload stop()s them first), but the UI's
    // query_status polls during Loading, when load_loop is building m_engine —
    // this mutex serializes that.  Locked once per window in the worker hot
    // path (~5s cadence) so the cost is negligible.
    mutable std::mutex                  m_engineMutex;
    std::unique_ptr<whisper_xpu::Engine> m_engine;             // owner path
    whisper_xpu::Engine*                 m_sharedEngine = nullptr;  // test path (borrowed)
    std::vector<whisper_state*>          m_states;          // POOL_SIZE, owned
    std::vector<std::string>             m_workerLang;      // POOL_SIZE, per-worker lang cache
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

    // Background model-load thread (owner path only).  Aborted by reload()/dtor
    // if still running.  m_loadGeneration lets a stale load_loop detect that a
    // newer reload started and exit early instead of racing the new setup.
    std::thread                 m_loadThread;
    std::atomic<bool>           m_loadAbort{false};
    std::atomic<unsigned>       m_loadGeneration{0};   // bumped by reload/async_setup
    std::atomic<bool>           m_loading{false};      // load_loop is running (for query_status)
    std::mutex                  m_launchMutex;          // guards launch_threads_if_ready + the flags below
    std::atomic<bool>           m_engineReady{false};   // warmup done
    std::atomic<bool>           m_captureActive{false}; // PortAudio open
    std::atomic<bool>           m_engineFailed{false};  // last load failed
    std::atomic<long>          m_totalChars{0};        // for query_status

    std::atomic<bool>           m_recording{false};
    std::atomic<bool>           m_stopping{false};  // aborts in-flight whisper_full + exits all loops

    // Introspection counters (populated across threads; read via stats()).
    std::atomic<int>  m_dispatched{0};
    std::atomic<int>  m_completed{0};
    std::atomic<int>  m_workersUsed{0};   // bitmask
    std::atomic<int>  m_segsKept{0};
    std::atomic<int>  m_segsDropped{0};
    std::atomic<int>  m_lastStopMs{0};
};
