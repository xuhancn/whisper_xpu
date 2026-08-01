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
    Draining,    // Stop clicked — draining buffered audio + in-flight transcribe (async_stop on a bg thread)
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
//   reload(m,d)   — stop() if recording, join the prior load thread (it has
//                  already finished — the UI blocks reload during Loading via
//                  can_reload()), drop the old Engine + states, then start a
//                  fresh load_loop.  The UI's sync thread observes Loading →
//                  Ready/Failed.  Never blocks (the join is instant because
//                  can_reload() guaranteed idle when the UI called this).
//   dtor          — join the load thread + pipeline threads, free engine.
//
// CRASH-PREVENTION MODEL: the UI (Settings OK handler) refuses to reload
// while a load is in progress (can_reload() == false ⇒ "still loading"
// messagebox).  This means reload() is NEVER called mid-warmup, so the load
// thread is never aborted/interrupted — GPU warmup runs to completion every
// time.  The scheduler thus needs NO abort flag, NO generation counter, NO
// discard-and-restart command buffer: the UI blocks the danger, the
// scheduler stays simple.
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

    // Immediate reload: stop any recording, join the (already-finished) load
    // thread, drop the old Engine + states, then start a fresh load_loop with
    // the new model/device.  The UI's sync thread observes Loading → Ready/
    // Failed.  Never blocks (the join is instant — the UI gates this on
    // can_reload(), so reload is never called mid-warmup).
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

    // Async stop: sets m_draining immediately (UI shows "Finishing…" via
    // query_status), runs stop() on a background thread so the GUI never
    // blocks during the drain.  The drain transcribes all buffered audio
    // + in-flight windows before shutting down.  The UI's sync thread polls
    // query_status() and sees Draining → Ready when done.  Record/Settings
    // are blocked while draining (can_reload() checks !m_draining).
    void async_stop();

    // ── Queries (pure data — the UI's sync thread polls this) ──

    bool is_recording() const { return m_recording.load(); }
    bool is_ready() const { return m_engineReady.load(); }
    // True iff the engine is ready AND no load is in progress — the UI gates
    // reload() on this so the load thread is never interrupted mid-warmup.
    // (m_engineReady covers the engine-built case; m_isLoading covers the
    // narrow window where warmup finished but reload hasn't reset it yet, and
    // vice-versa.)  The Settings OK handler blocks reload unless this is true.
    bool can_reload() const { return m_engineReady.load() && !m_isLoading.load() && !m_draining.load(); }
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
        std::string     prompt;              // prev emitted text → whisper initial_prompt (context stitching)
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

    // Boundary post-processing: if the last N words of prevTail equal the
    // first N words of curHead (N up to ~3), drop the head copy — a word
    // straddling a 5s window seam can otherwise be transcribed twice (the
    // midpoint dedup catches most, this is the cleanup net).  Applied at
    // chunk seams only, not within a chunk.
    static std::string trim_overlap_repeat(const std::string& prevTail,
                                           const std::string& curHead);

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
    // speech.  No abort flag — warmup always runs to completion (the UI blocks
    // reload during Loading, so it's never interrupted).
    void warmup_states();

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
    // path) + N per-worker whisper_state*s (own KV cache + compute buffers),
    // created in start()/start_no_capture()/load_loop(), freed in stop().
    // POOL_SIZE is runtime-decided in init_states(): GPU=1, CPU=4.  WHY: on
    // the GPU, all worker states' whisper_full_with_state calls go through the
    // SAME SYCL default_queue (ggml-sycl's stream() returns the device's
    // singleton default queue).  sycl::queue is NOT thread-safe for concurrent
    // submit from multiple host threads → async SYCL error →
    // ggml_backend_sycl_synchronize aborts (c0000409) or AVs (c0000005 reading
    // 0xFFFF...FFFF).  One GPU worker serializes queue access with zero locks
    // (the pool is just 1).  CPU is fine with 4 workers (ggml-cpu is
    // thread-safe / thread-pooled per call).  See memory
    // [[ze-driver-crash-under-debugger]] (root cause: shared SYCL queue).
    static constexpr int POOL_SIZE_CPU = 4;
    static constexpr int POOL_SIZE_GPU = 1;
    int m_poolSize = POOL_SIZE_CPU;   // set in init_states() per engine
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
    // Context + paragraph state (merger owns; under m_mergerMutex).
    // m_lastEmittedText: running transcript tail (capped ~200 chars) fed as
    //   initial_prompt to the next window via WindowJob::prompt (context
    //   stitching).  Snapshot is read by the windower (best-effort).
    // m_lastSegGlobalEndMs: global end-ms of the last KEPT segment, for gap-
    //   based paragraph detection (gap<1s same para, >2s new para, 1-2s soft).
    std::string                 m_lastEmittedText;
    int64_t                     m_lastSegGlobalEndMs = -1;

    std::thread                 m_windowerThread;
    std::vector<std::thread>    m_workerThreads;
    std::thread                 m_mergerThread;

    // Background model-load thread (owner path only).  Joined by reload()
    // (after the UI confirmed idle via can_reload()) and by the dtor.  NEVER
    // aborted/interrupted — the UI blocks reload during Loading, so warmup
    // always runs to completion (no abort flag, no generation counter).
    std::thread                 m_loadThread;
    std::thread                 m_stopThread;             // async_stop's drain thread (joined in dtor)
    std::atomic<bool>           m_loading{false};      // load_loop is running (for query_status + can_reload)
    std::atomic<bool>           m_isLoading{false};    // reload/start gate — true while a load is in flight
    std::mutex                  m_launchMutex;          // guards launch_threads_if_ready + the flags below
    std::atomic<bool>           m_engineReady{false};   // warmup done
    std::atomic<bool>           m_captureActive{false}; // PortAudio open
    std::atomic<bool>           m_engineFailed{false};  // last load failed
    std::atomic<long>          m_totalChars{0};        // for query_status

    std::atomic<bool>           m_recording{false};
    std::atomic<bool>           m_stopping{false};  // genuine shutdown: aborts in-flight whisper_full + exits all loops (dtor/reload)
    std::atomic<bool>           m_draining{false};  // stop() drain phase: windower drains remaining ring, workers finish in-flight, merger flushes — do NOT abort in-flight transcribe

    // Introspection counters (populated across threads; read via stats()).
    std::atomic<int>  m_dispatched{0};
    std::atomic<int>  m_completed{0};
    std::atomic<int>  m_workersUsed{0};   // bitmask
    std::atomic<int>  m_segsKept{0};
    std::atomic<int>  m_segsDropped{0};
    std::atomic<int>  m_lastStopMs{0};
};
