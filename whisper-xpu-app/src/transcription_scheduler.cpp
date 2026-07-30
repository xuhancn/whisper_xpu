#include "src/transcription_scheduler.h"
#include "src/sched_log.h"
#include "../audio_capture.h"
#include "engine.h"
#include "device_detect.h"   // kDeviceCPU / kDeviceAuto + get_device_info

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

// ── Pipeline constants (16 kHz mono) ──
// WINDOW_NEW_S=5: the verified accuracy baseline.  warmup_states() now
// removes the first-window cold-start stall (the JIT/buffer/lang slow paths),
// so we no longer need a smaller window to mask it; keep 5s for correct
// accuracy.  First-word latency is ~5s + compute, acceptable and stable.
static constexpr int SR             = 16000;
static constexpr int WINDOW_NEW_S   = 5;                          // new audio per window
static constexpr int TAIL_S         = 1;                          // overlap tail from prev window
static constexpr int CADENCE_MS     = WINDOW_NEW_S * 1000;        // 5000 — one window per 5s
static constexpr int CORE_GUARD_MS  = 250;                        // boundary dedup guard band

static constexpr size_t NEW_SAMPLES  = (size_t)SR * WINDOW_NEW_S;        // 80000  (5s)
static constexpr size_t TAIL_SAMPLES  = (size_t)SR * TAIL_S;            // 16000  (1s)
static constexpr size_t TOTAL_SAMPLES = NEW_SAMPLES + TAIL_SAMPLES;     // 96000  (6s)

static int now_ms() {
    return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::join_thread(std::thread& t) {
    if (t.joinable()) t.join();
}

// Static PortAudio callback → pushes to ring buffer
void TranscriptionScheduler::on_audio_cb(TranscriptionScheduler* self,
                                         const float* samples, size_t count) {
    self->m_ring.push(samples, count);
}

// ────────────────────────────────────────────────────────────
// Constructor / Destructor
// ────────────────────────────────────────────────────────────

TranscriptionScheduler::TranscriptionScheduler(TextCallback on_text, size_t ring_capacity)
    : m_ownsEngine(false)   // test path: engine borrowed via start_no_capture
    , m_on_text(std::move(on_text))
    , m_ring(ring_capacity ? ring_capacity : (size_t)(20 * SR))
{
    sched_log("[Scheduler] created (test path; ring=%zu samples)", m_ring.capacity());
    // No background load thread — the test owns and borrows its own Engine.
}

TranscriptionScheduler::TranscriptionScheduler(TextCallback on_text,
                                               std::string model_path,
                                               int device_index,
                                               size_t ring_capacity)
    : m_ownsEngine(true)
    , m_modelPath(std::move(model_path))
    , m_deviceIndex(device_index)
    , m_on_text(std::move(on_text))
    , m_ring(ring_capacity ? ring_capacity : (size_t)(20 * SR))
{
    sched_log("[Scheduler] created (owner path; model='%s' dev=%d ring=%zu samples)",
              m_modelPath.c_str(), m_deviceIndex, m_ring.capacity());
    // Kick off the background model load immediately (never blocks the GUI).
    if (!m_modelPath.empty()) {
        async_setup(m_modelPath, m_deviceIndex);
    } else {
        // No model configured yet — leave the engine null; the app (Settings)
        // will surface "no model".  is_ready() stays false.
        sched_log("[Scheduler] no model path — load thread not started");
    }
}

TranscriptionScheduler::~TranscriptionScheduler() {
    if (m_recording) stop();
    // Join the background load thread if still running (e.g. the app closed
    // during the startup load).  We do NOT abort it — there's no abort flag
    // anymore (the UI blocks reload during Loading, so warmup is never
    // interrupted).  The join blocks until load_loop() finishes naturally.
    // Closing mid-load therefore waits for the full warmup (~15s on GPU turbo)
    // rather than crashing; a stuck SYCL JIT would hang here, but that's a
    // driver fault, not something an abort could fix safely.
    if (m_ownsEngine && m_loadThread.joinable()) {
        sched_log("[Scheduler] dtor: joining load thread...");
        m_loadThread.join();
        sched_log("[Scheduler] dtor: load thread joined");
    }
    join_thread(m_windowerThread);
    for (auto& t : m_workerThreads) join_thread(t);
    join_thread(m_mergerThread);
    // Free any per-worker states left by an aborted-mid-init load_loop (it
    // publishes m_engine + populates m_states before warmup; the dtor must
    // free them against the owned engine BEFORE the unique_ptr releases it).
    {
        std::lock_guard<std::mutex> lk(m_engineMutex);
        whisper_xpu::Engine* eng = m_engine.get();
        if (eng) for (whisper_state* st : m_states) eng->free_state(st);
        m_states.clear();
        m_workerLang.clear();
    }
    sched_log("[Scheduler] destroyed");
}

// ────────────────────────────────────────────────────────────
// Active engine accessor (owned in the app path, borrowed in tests)
// ────────────────────────────────────────────────────────────

whisper_xpu::Engine* TranscriptionScheduler::engine() const {
    // m_engine is the owned engine (app path); m_sharedEngine is borrowed
    // (test path via start_no_capture).  Exactly one is set at any time.
    // Guarded: reload()/dtor mutate these while the UI's query_status and the
    // worker threads read them.  Workers run only while m_engineReady (reload
    // stop()s them first), but query_status polls during Loading.
    std::lock_guard<std::mutex> lk(m_engineMutex);
    return m_engine ? m_engine.get() : m_sharedEngine;
}

// ────────────────────────────────────────────────────────────
// Per-worker state creation (shared by start / start_no_capture / load_loop)
// ────────────────────────────────────────────────────────────────────

bool TranscriptionScheduler::init_states() {
    whisper_xpu::Engine* eng = engine();
    if (!eng) {
        sched_log("[Scheduler] init_states: no engine");
        return false;
    }
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 1) hw = 4;
    m_nThreadsPerWorker = std::max(1, hw / POOL_SIZE);

    m_states.clear();
    m_workerLang.clear();
    m_states.reserve(POOL_SIZE);
    m_workerLang.resize(POOL_SIZE);  // empty ⇒ first window auto-detects
    for (int i = 0; i < POOL_SIZE; ++i) {
        whisper_state* st = eng->create_state();
        if (!st) {
            sched_log("[Scheduler] state %d create failed — freeing partial", i);
            for (whisper_state* s : m_states) eng->free_state(s);
            m_states.clear();
            m_workerLang.clear();
            return false;
        }
        m_states.push_back(st);
    }
    sched_log("[Scheduler] pool: 1 Engine + %d states × %d threads = %d total (hw=%d)",
              POOL_SIZE, m_nThreadsPerWorker, POOL_SIZE * m_nThreadsPerWorker, hw);
    return true;
}

// ────────────────────────────────────────────────────────────
// Per-worker state warmup (called before threads start)
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::warmup_states() {
    // 1s of silence @ 16 kHz — enough to trigger one full encoder+decoder
    // pass through whisper_full_with_state on each worker's own state, priming
    // the SYCL/Level Zero first-kernel JIT compile + per-state compute-buffer
    // allocation BEFORE any real audio is queued.
    //
    // IMPORTANT: the language detection pass is primed too, but we deliberately
    // use a THROWAWAY local `lang` (empty ⇒ auto-detect) and DO NOT write the
    // result back into m_workerLang[i].  whisper auto-detecting on silence
    // returns a default (typically "en") which, if cached, would pin every
    // real Chinese window to English ⇒ garbage output.  Leaving m_workerLang[i]
    // empty means the real first window still auto-detects on real speech; that
    // detection is cheap now because the JIT/encoder cold path was already
    // primed here.
    //
    // No abort flag — warmup always runs to completion.  The UI blocks reload
    // during Loading (can_reload()), so this is never interrupted mid-pass.
    constexpr int kWarmupN = 1 * SR;
    std::vector<float> silence(kWarmupN, 0.0f);

    whisper_xpu::Engine* eng = engine();
    for (int i = 0; i < POOL_SIZE; ++i) {
        if (!m_states[i] || !eng) continue;
        std::string lang;  // throwaway — auto-detects on silence, result
                           // discarded; m_workerLang[i] stays empty.
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = eng->transcribe_window_with_state(
            m_states[i], m_nThreadsPerWorker, lang,
            silence.data(), (int)silence.size(), nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        sched_log("[Scheduler] warmup state %d OK (%.0fms, %zu segs) — lang left for real audio",
                  i, ms, r.segments.size());
    }
    sched_log("[Scheduler] warmup done — JIT/buffer primed for all %d states",
              POOL_SIZE);
}

// ────────────────────────────────────────────────────────────
// Background model load (owner path — runs once/reload, off the GUI thread)
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::load_loop() {
    // SIMPLE load: Engine ctor → init_states → warmup → ready → launch → exit.
    // No loop, no abort checks, no generation bumps.  The UI blocks reload
    // during Loading (can_reload()), so this load is NEVER interrupted —
    // GPU warmup always runs to completion.  That's the whole crash fix.
    struct LoadingGuard {
        std::atomic<bool>& loading;
        std::atomic<bool>& isLoading;
        ~LoadingGuard() { loading.store(false); isLoading.store(false); }
    } guard{m_loading, m_isLoading};
    m_loading.store(true);    // query_status() reports Loading while we run
    m_isLoading.store(true);  // can_reload() = false while we run
    sched_log("[Scheduler] load_loop: constructing Engine (model='%s' dev=%d)...",
              m_modelPath.c_str(), m_deviceIndex);

    // The Engine ctor (model load + ggml alloc) lives inside the icpx-built
    // core DLL, which SEH-guards any SYCL AV.  A try/catch (MSVC caller) turns
    // a hard crash into a logged failure so the app stays alive.  Build into a
    // local first (the ctor is slow), then publish under m_engineMutex so the
    // UI's query_status sees a stable pointer.
    bool failed = false;
    std::unique_ptr<whisper_xpu::Engine> built;
    try {
        built = std::make_unique<whisper_xpu::Engine>(m_modelPath, m_deviceIndex);
    } catch (const std::exception& e) {
        sched_log("[Scheduler] load_loop: Engine ctor FAILED: %s", e.what());
        failed = true;
    }
    {
        std::lock_guard<std::mutex> lk(m_engineMutex);
        m_engine = std::move(built);
    }
    if (failed || !m_engine) {
        m_engineFailed.store(true);
        return;
    }

    // Create the 4 per-worker states on the owned engine, then warm up each
    // (SYCL JIT + buffer alloc).  This is the slow (~10-25s) part but it runs
    // here, never on the GUI thread.  No pipeline workers exist yet, so the
    // GPU is touched by exactly one thread — safe.
    if (!init_states()) {
        sched_log("[Scheduler] load_loop: init_states failed");
        m_engineFailed.store(true);
        return;
    }
    warmup_states();

    // Engine + states + JIT are all primed → mark ready and, if Record was
    // already pressed (capture active), launch the pipeline threads now.
    {
        std::lock_guard<std::mutex> lk(m_launchMutex);
        m_engineReady.store(true);
    }
    sched_log("[Scheduler] load_loop: engine ready");
    launch_threads_if_ready();
}

bool TranscriptionScheduler::async_setup(const std::string& model_path, int device_index) {
    if (model_path.empty()) {
        sched_log("[Scheduler] async_setup: empty model path — not loading");
        return false;
    }
    m_modelPath = model_path;
    m_deviceIndex = device_index;
    m_engineFailed.store(false);
    // Cache the basename for query_status (read by the sync thread under
    // m_engineMutex; computing it here on the wx thread keeps the read tear-free).
    {
        std::lock_guard<std::mutex> lk(m_engineMutex);
        if (m_modelPath.empty()) m_modelName.clear();
        else {
            auto pos = m_modelPath.find_last_of("\\/");
            m_modelName = (pos == std::string::npos) ? m_modelPath
                                                     : m_modelPath.substr(pos + 1);
        }
    }
    // Start the load on a fresh background thread.  No abort, no generation —
    // the caller (ctor/reload) guarantees no load is in flight (the UI gates
    // reload on can_reload(); async_setup from the ctor is the first load).
    m_loadThread = std::thread(&TranscriptionScheduler::load_loop, this);
    sched_log("[Scheduler] async_setup: started (model='%s' dev=%d)",
              m_modelPath.c_str(), m_deviceIndex);
    return true;
}

void TranscriptionScheduler::reload(const std::string& model_path, int device_index) {
    sched_log("[Scheduler] reload: model='%s' dev=%d", model_path.c_str(), device_index);
    // 1. Stop any recording (joins the pipeline threads + frees the states).
    if (m_recording) stop();
    // Set the loading gate NOW (before teardown) so can_reload() stays false
    // for the whole reload — the UI can't sneak a second reload into the
    // window between dropping the old engine and load_loop setting it.
    m_isLoading.store(true);
    m_loading.store(true);
    // 2. Join the prior load thread.  The UI guaranteed idle via can_reload(),
    //    so the thread has already finished — the join is instant.  (Defensive:
    //    if it's somehow still joinable, block until it exits; never start a
    //    second concurrent load_loop — that's the ggml-sycl AV root cause.)
    if (m_loadThread.joinable()) m_loadThread.join();
    // 3. Drop the old Engine + states + flags under m_engineMutex so the WHOLE
    //    teardown is atomic vs query_status/workers.  (States were freed in
    //    stop() if we were recording; if we were just idle-but-ready, free
    //    them now on the caller thread — safe: no pipeline/workers running.)
    {
        std::lock_guard<std::mutex> lk(m_engineMutex);
        whisper_xpu::Engine* eng = m_engine.get();
        if (eng) for (whisper_state* st : m_states) eng->free_state(st);
        m_states.clear();
        m_workerLang.clear();
        m_engine.reset();
        m_engineReady.store(false);
        m_engineFailed.store(false);
        m_captureActive.store(false);
        m_stopping.store(false);
    }
    // 4. Start the new setup on a fresh background thread (sets m_isLoading).
    m_modelPath = model_path;
    m_deviceIndex = device_index;
    async_setup(model_path, device_index);
}

// ────────────────────────────────────────────────────────────
// start (app path — capture now, pipeline when ready)
// ────────────────────────────────────────────────────────────

bool TranscriptionScheduler::start(int micIndex) {
    if (m_recording) return true;

    m_stopping = false;
    m_nextEmit = 0;
    m_pending.clear();
    m_workerQueue.clear();
    m_mergerQueue.clear();
    m_ring.clear();
    m_dispatched = m_completed = m_workersUsed = 0;
    m_segsKept = m_segsDropped = 0; m_lastStopMs = 0;
    m_totalChars.store(0);   // per-session char count (stats().total_chars + query_status)

    // Open PortAudio FIRST — it's model-independent and fast (<100ms).  Audio
    // flows into the ring immediately so no early speech is lost while the
    // background load finishes.  If the load isn't done yet, the captured PCM
    // buffers in the 20s ring until launch_threads_if_ready() spawns workers.
    m_capture = std::make_unique<AudioCapture>();
    m_capture->set_callback([this](const float* s, size_t n) -> size_t {
        on_audio_cb(this, s, n);
        return n;
    });
    bool ok = m_capture->start(micIndex, SR, 512);
    sched_log("[Scheduler] start mic=%d ok=%d (engineReady=%d)",
              micIndex, (int)ok, (int)m_engineReady.load());
    if (!ok) {
        m_capture.reset();
        return false;
    }

    m_recording = true;
    {
        std::lock_guard<std::mutex> lk(m_launchMutex);
        m_captureActive.store(true);
    }
    // If the engine is already loaded + warmed, launch the pipeline now
    // (the fast, every-Record-after-the-first path).  Otherwise load_loop()
    // will call this once it finishes.
    launch_threads_if_ready();
    return true;
}

bool TranscriptionScheduler::start_no_capture(whisper_xpu::Engine& sharedEngine) {
    if (m_recording) return true;
    // Headless/test path: the caller owns the Engine and borrows it.  No
    // background load thread, no AudioCapture.  m_ownsEngine stays false.
    m_sharedEngine = &sharedEngine;

    m_stopping = false;
    m_nextEmit = 0;
    m_pending.clear();
    m_workerQueue.clear();
    m_mergerQueue.clear();
    m_ring.clear();
    m_dispatched = m_completed = m_workersUsed = 0;
    m_segsKept = m_segsDropped = 0; m_lastStopMs = 0;
    m_totalChars.store(0);

    if (!init_states()) { m_sharedEngine = nullptr; return false; }

    // Same warmup as the app path — prime JIT/buffer/lang so file-replay
    // workers don't backlog behind worker 0's first-window slow paths.
    warmup_states();

    // No AudioCapture — caller feeds the ring via feed_audio().  Log so it's
    // clear in headless runs that the mic was intentionally skipped.
    sched_log("[Scheduler] start_no_capture (headless; feed via feed_audio)");

    m_recording = true;
    {
        std::lock_guard<std::mutex> lk(m_launchMutex);
        m_engineReady.store(true);   // borrowed engine is ready by construction
        // (no capture in headless mode; launch directly below)
    }
    m_windowerThread = std::thread(&TranscriptionScheduler::windower_loop, this);
    m_workerThreads.clear();
    m_workerThreads.reserve(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i)
        m_workerThreads.emplace_back(&TranscriptionScheduler::worker_loop, this, i);
    m_mergerThread = std::thread(&TranscriptionScheduler::merger_loop, this);
    return true;
}

// ────────────────────────────────────────────────────────────
// launch_threads_if_ready — rendezvous: spawn pipeline threads once both
// the engine is ready AND capture is active.  Idempotent + guarded.
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::launch_threads_if_ready() {
    // Headless path already launched its threads inside start_no_capture;
    // m_captureActive is false there, so this guard is a no-op for tests.
    if (!m_recording.load()) return;
    bool launch = false;
    {
        std::lock_guard<std::mutex> lk(m_launchMutex);
        if (m_engineReady.load() && m_captureActive.load() && m_windowerThread.get_id() == std::thread::id()) {
            launch = true;
        }
    }
    if (!launch) return;

    // The per-worker states may have been freed by a previous stop() (the
    // owned Engine survives stop() for reuse, but states don't).  Recreate
    // them now if needed — cheap vs. the model load (just 4 KV-cache allocs);
    // the SYCL JIT is already cached from warmup_states() so no re-prime.
    if (m_states.empty()) {
        if (!init_states()) {
            sched_log("[Scheduler] launch_threads_if_ready: init_states failed — not launching");
            return;
        }
    }

    // Only launch once: the m_windowerThread.get_id()==id check above (under
    // the lock) guarantees a single caller wins.  Capture is open, the engine
    // is primed — spin the windower/worker/merger threads to drain the ring.
    sched_log("[Scheduler] launching pipeline threads (engine ready + capture active)");
    m_windowerThread = std::thread(&TranscriptionScheduler::windower_loop, this);
    m_workerThreads.clear();
    m_workerThreads.reserve(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i)
        m_workerThreads.emplace_back(&TranscriptionScheduler::worker_loop, this, i);
    m_mergerThread = std::thread(&TranscriptionScheduler::merger_loop, this);
}

void TranscriptionScheduler::feed_audio(const float* samples, size_t count) {
    m_ring.push(samples, count);
}

void TranscriptionScheduler::stop() {
    if (!m_recording) return;

    int t0 = now_ms();
    sched_log("[Scheduler] stop");
    m_recording = false;
    m_stopping  = true;   // aborts in-flight transcribe_window + exits all loops

    m_workerCv.notify_all();   // wake workers blocked on the queue
    m_mergerCv.notify_all();   // wake merger

    // Join all threads before freeing the per-worker states so none touch
    // `this` / m_states / the borrowed Engine after stop.  In-flight
    // transcribe_window_with_state aborts in ms via the abort flag.
    join_thread(m_windowerThread);
    for (auto& t : m_workerThreads) join_thread(t);
    join_thread(m_mergerThread);
    m_workerThreads.clear();

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }
    // Free the per-worker states; the Engine itself survives stop() (owned:
    // reused for the next Record; borrowed: owned by the test caller).  States
    // must be freed only after workers have joined — done above — so none is
    // mid-whisper_full_with_state.
    whisper_xpu::Engine* eng = engine();
    if (eng) {
        for (whisper_state* st : m_states) eng->free_state(st);
    }
    m_states.clear();
    m_workerLang.clear();
    // Borrowed-engine pointer (test path) is cleared so a later owner-path
    // start() doesn't read a dangling m_sharedEngine; the owned m_engine and
    // m_engineReady flag are NOT reset — the engine stays ready for reuse.
    m_sharedEngine = nullptr;
    m_captureActive.store(false);
    m_lastStopMs = now_ms() - t0;
    sched_log("[Scheduler] stopped (%dms)", m_lastStopMs.load());
}

// ────────────────────────────────────────────────────────────
// Windower: ring → 6s windows (1s tail + 5s new) @ 5s cadence → worker queue
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::windower_loop() {
    sched_log("[Scheduler] windower started");

    // window 0's tail is 1s of silence (no previous window) — keeps every
    // window a uniform 6s with a 1s prefix, so the merger's core band is a
    // constant local [1250,5750]ms (global = pcm_start_global + local).
    std::vector<float> prevTail(TAIL_SAMPLES, 0.0f);
    std::vector<float> newBuf(NEW_SAMPLES);
    int chunkIndex = 0;

    while (m_recording) {
        // Gather 5s of new audio.  Blocks until a full 5s arrives (sleeping
        // when the ring is empty) — this is what paces the pipeline to real
        // audio time in the mic path.  In file-replay, the caller pre-loaded
        // the ring so this drains as fast as the workers keep up.
        size_t got = 0;
        while (got < NEW_SAMPLES && m_recording) {
            size_t n = m_ring.pull(newBuf.data() + got, NEW_SAMPLES - got);
            got += n;
            if (n == 0) std::this_thread::sleep_for(20ms);
        }
        if (!m_recording) break;

        // 6s window = [prevTail (1s) || new (5s)].
        std::vector<float> pcm;
        pcm.reserve(TOTAL_SAMPLES);
        pcm.insert(pcm.end(), prevTail.begin(), prevTail.end());
        pcm.insert(pcm.end(), newBuf.begin(), newBuf.end());

        // Next window's tail = last 1s of THIS window's new audio (the audio
        // straddling the next 5s boundary).
        prevTail.assign(newBuf.data() + (NEW_SAMPLES - TAIL_SAMPLES),
                        newBuf.data() +  NEW_SAMPLES);

        const int64_t pcm_start_global_ms =
            (int64_t)chunkIndex * CADENCE_MS - (int64_t)TAIL_S * 1000;

        size_t queued;
        {
            std::lock_guard<std::mutex> lk(m_workerMutex);
            m_workerQueue.push_back(WindowJob{chunkIndex, pcm_start_global_ms, std::move(pcm)});
            queued = m_workerQueue.size();
        }
        m_workerCv.notify_one();
        m_dispatched.fetch_add(1);

        sched_log("[Scheduler] window %d queued: %zu samples start=%lldms ring=%zu wq=%zu",
                  chunkIndex, TOTAL_SAMPLES, (long long)pcm_start_global_ms,
                  m_ring.available(), queued);
        chunkIndex++;
    }

    m_workerCv.notify_all();
    sched_log("[Scheduler] windower exiting (dispatched=%d)",
              m_dispatched.load());
}

// ────────────────────────────────────────────────────────────
// Worker: pop window → Engine::transcribe_window → post result to merger
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::worker_loop(int workerId) {
    sched_log("[Scheduler] worker %d started (n_threads=%d)", workerId, m_nThreadsPerWorker);

    while (true) {
        WindowJob job;
        {
            std::unique_lock<std::mutex> lk(m_workerMutex);
            m_workerCv.wait(lk, [this] {
                return m_stopping.load() || !m_workerQueue.empty();
            });
            if (m_stopping.load() && m_workerQueue.empty()) break;
            if (m_workerQueue.empty()) continue;   // spurious wake
            job = std::move(m_workerQueue.front());
            m_workerQueue.pop_front();
        }
        m_workersUsed.fetch_or(1 << workerId);

        auto r = engine()->transcribe_window_with_state(
            m_states[workerId], m_nThreadsPerWorker, m_workerLang[workerId],
            job.pcm.data(), (int)job.pcm.size(), &m_stopping);
        const size_t segs  = r.segments.size();
        const double pms   = r.processing_time_ms;
        const bool   aborted = r.aborted;

        if (aborted) {
            r.segments.clear();
            sched_log("[Scheduler] worker %d window %d ABORTED (%.0fms) — posting empty",
                      workerId, job.index, pms);
        }

        {
            std::lock_guard<std::mutex> lk(m_mergerMutex);
            m_mergerQueue.push_back(ChunkDone{job.index, job.pcm_start_global_ms, std::move(r)});
        }
        m_mergerCv.notify_one();
        m_completed.fetch_add(1);

        if (!aborted)
            sched_log("[Scheduler] worker %d window %d: %zu segs %.0fms → merger",
                      workerId, job.index, segs, pms);
    }
    sched_log("[Scheduler] worker %d exiting", workerId);
}

// ────────────────────────────────────────────────────────────
// Merger: in-order emit + overlap dedup by global midpoint
// ────────────────────────────────────────────────────────────

void TranscriptionScheduler::merger_loop() {
    sched_log("[Scheduler] merger started");

    while (true) {
        ChunkDone done;
        {
            std::unique_lock<std::mutex> lk(m_mergerMutex);
            m_mergerCv.wait(lk, [this] {
                return m_stopping.load() || !m_mergerQueue.empty();
            });
            if (m_stopping.load() && m_mergerQueue.empty()) break;
            if (m_mergerQueue.empty()) continue;
            done = std::move(m_mergerQueue.front());
            m_mergerQueue.pop_front();
            // Stash under the lock: stats() polls m_pending from another thread
            // and a concurrent std::map insert would race (not node-stable).
            m_pending[done.index] = std::move(done);
        }

        // Drain from m_nextEmit upward — all map/nextEmit access is under the
        // lock so stats() sees a consistent snapshot.  The per-chunk emit
        // (on_text/on_chunk, possibly slow/re-entrant) runs OUTSIDE the lock.
        while (true) {
            ChunkDone c;
            {
                std::lock_guard<std::mutex> lk(m_mergerMutex);
                auto it = m_pending.find(m_nextEmit);
                if (it == m_pending.end()) {
                    if (!m_stopping.load())
                        sched_log("[Scheduler] merger holding for chunk %d", m_nextEmit);
                    break;   // wait for the in-order predecessor
                }
                c = std::move(it->second);
                m_pending.erase(it);
                m_nextEmit++;
            }

            // Emit chunk (m_nextEmit-1): keep segments whose GLOBAL midpoint
            // lands in the core band [K*5000+250, K*5000+4750]ms.  The 0.25s
            // guards at each 5s boundary drop the overlap region shared with
            // the neighbour window ⇒ no word emitted twice, no word lost.
            const int idx = c.index;
            const int64_t kStart = (int64_t)idx * CADENCE_MS;
            const int64_t lo = kStart + CORE_GUARD_MS;
            const int64_t hi = kStart + CADENCE_MS - CORE_GUARD_MS;

            std::string text;
            int kept = 0;
            for (const auto& seg : c.result.segments) {
                const int64_t mid = c.pcm_start_global_ms + (seg.t0_ms + seg.t1_ms) / 2;
                const bool inBand = (mid >= lo && mid <= hi);
                std::string t = seg.text;
                const size_t s = t.find_first_not_of(' ');
                if (s != std::string::npos) t.erase(0, s);
                if (!inBand) {
                    m_segsDropped.fetch_add(1);
                    sched_log("[Scheduler]   drop seg: local[%lld,%lld]ms global_mid=%lldms band=[%lld,%lld] '%.40s'",
                              (long long)seg.t0_ms, (long long)seg.t1_ms,
                              (long long)mid, (long long)lo, (long long)hi, t.c_str());
                    continue;
                }
                if (t.empty()) continue;
                text += t;
                text += ' ';
                ++kept;
                m_segsKept.fetch_add(1);
            }
            if (!text.empty() && text.back() == ' ') text.pop_back();

            if (!text.empty()) {
                m_totalChars.fetch_add((long)text.size());   // stats + query_status
                if (m_on_text) m_on_text(text);
            }
            if (m_on_chunk) m_on_chunk(idx, text);
            sched_log("[Scheduler] emit chunk %d: %d/%zu segs kept, %zu chars",
                      idx, kept, c.result.segments.size(), text.size());
        }
    }
    sched_log("[Scheduler] merger exiting (pending=%zu, nextEmit=%d)",
              m_pending.size(), m_nextEmit);
}

// ────────────────────────────────────────────────────────────
// Introspection
// ────────────────────────────────────────────────────────────

TranscriptionScheduler::PipelineStats TranscriptionScheduler::stats() const {
    PipelineStats s;
    s.windows_dispatched = m_dispatched.load();
    s.windows_completed  = m_completed.load();
    s.workers_used_mask  = m_workersUsed.load();
    s.segments_kept      = m_segsKept.load();
    s.segments_dropped   = m_segsDropped.load();
    s.total_chars        = m_totalChars.load();
    s.last_stop_ms       = m_lastStopMs.load();
    {
        std::lock_guard<std::mutex> lk(m_mergerMutex);
        s.next_emit = m_nextEmit;
        s.pending   = (int)m_pending.size();
    }
    return s;
}

SchedulerStatus TranscriptionScheduler::query_status() const {
    // Pure data snapshot — the UI's sync thread calls this every 100ms.  State
    // precedence: Recording > Loading > Failed > Ready > Idle.  The exact
    // field set is compared by SchedulerStatus::operator==, so no-op polls
    // don't trigger a UI refresh.
    SchedulerStatus s;
    s.recording   = m_recording.load();
    s.gpu_ready   = m_engineReady.load();
    s.total_chars = m_totalChars.load();
    // device_desc: owned engine (app path) or borrowed (test path).  Empty
    // before the engine is constructed.  Guard against reload() mutating it.
    {
        std::lock_guard<std::mutex> lk(m_engineMutex);
        if (m_engine)             s.device_desc = m_engine->device_description();
        else if (m_sharedEngine)  s.device_desc = m_sharedEngine->device_description();
        s.model_name = m_modelName;   // cached basename (set in async_setup)
    }

    if (m_recording.load()) {
        s.state = SchedulerState::Recording;
    } else if (m_loading.load() && !m_engineReady.load() && !m_engineFailed.load()) {
        // load_loop is running and hasn't reached ready/failed yet.
        s.state = SchedulerState::Loading;
    } else if (m_engineFailed.load()) {
        s.state = SchedulerState::Failed;
    } else if (m_engineReady.load()) {
        s.state = SchedulerState::Ready;
    } else {
        s.state = SchedulerState::Idle;
    }
    return s;
}

