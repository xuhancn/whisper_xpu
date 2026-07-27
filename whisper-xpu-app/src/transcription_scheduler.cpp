#include "src/transcription_scheduler.h"
#include "src/sched_log.h"
#include "../audio_capture.h"
#include "engine.h"

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

// ── Pipeline constants (16 kHz mono) ──
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
    : m_on_text(std::move(on_text))
    , m_ring(ring_capacity ? ring_capacity : (size_t)(10 * SR))
{
    sched_log("[Scheduler] created (ring=%zu samples)", m_ring.capacity());
}

TranscriptionScheduler::~TranscriptionScheduler() {
    if (m_recording) stop();
    join_thread(m_windowerThread);
    for (auto& t : m_workerThreads) join_thread(t);
    join_thread(m_mergerThread);
    sched_log("[Scheduler] destroyed");
}

// ────────────────────────────────────────────────────────────
// Pool creation (shared by start / start_no_capture)
// ────────────────────────────────────────────────────────────

bool TranscriptionScheduler::create_pool(const std::string& modelPath, int deviceIndex) {
    int hw = (int)std::thread::hardware_concurrency();
    if (hw < 1) hw = 4;
    m_nThreadsPerWorker = std::max(1, hw / POOL_SIZE);

    m_engines.clear();
    m_engines.reserve(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i) {
        try {
            m_engines.emplace_back(std::make_unique<whisper_xpu::Engine>(
                modelPath, deviceIndex, m_nThreadsPerWorker));
        } catch (const std::exception& e) {
            sched_log("[Scheduler] engine %d create failed: %s", i, e.what());
            m_engines.clear();
            return false;
        }
    }
    sched_log("[Scheduler] pool: %d engines × %d threads = %d total (hw=%d)",
              POOL_SIZE, m_nThreadsPerWorker, POOL_SIZE * m_nThreadsPerWorker, hw);
    return true;
}

// ────────────────────────────────────────────────────────────
// Start (app / mic) and start_no_capture (headless / file replay)
// ────────────────────────────────────────────────────────────

bool TranscriptionScheduler::start(int micIndex, const std::string& modelPath,
                                   int deviceIndex) {
    if (m_recording) return true;
    if (modelPath.empty()) return false;

    m_stopping = false;
    m_nextEmit = 0;
    m_pending.clear();
    m_workerQueue.clear();
    m_mergerQueue.clear();
    m_ring.clear();
    m_dispatched = m_completed = m_workersUsed = 0;
    m_segsKept = m_segsDropped = 0; m_charsEmitted = 0; m_lastStopMs = 0;

    if (!create_pool(modelPath, deviceIndex)) return false;

    m_capture = std::make_unique<AudioCapture>();
    m_capture->set_callback([this](const float* s, size_t n) -> size_t {
        on_audio_cb(this, s, n);
        return n;
    });
    bool ok = m_capture->start(micIndex, SR, 512);
    sched_log("[Scheduler] start mic=%d ok=%d", micIndex, (int)ok);
    if (!ok) { m_capture.reset(); m_engines.clear(); return false; }

    m_recording = true;
    m_windowerThread = std::thread(&TranscriptionScheduler::windower_loop, this);
    m_workerThreads.clear();
    m_workerThreads.reserve(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i)
        m_workerThreads.emplace_back(&TranscriptionScheduler::worker_loop, this, i);
    m_mergerThread = std::thread(&TranscriptionScheduler::merger_loop, this);
    return true;
}

bool TranscriptionScheduler::start_no_capture(const std::string& modelPath,
                                              int deviceIndex) {
    if (m_recording) return true;
    if (modelPath.empty()) return false;

    m_stopping = false;
    m_nextEmit = 0;
    m_pending.clear();
    m_workerQueue.clear();
    m_mergerQueue.clear();
    m_ring.clear();
    m_dispatched = m_completed = m_workersUsed = 0;
    m_segsKept = m_segsDropped = 0; m_charsEmitted = 0; m_lastStopMs = 0;

    if (!create_pool(modelPath, deviceIndex)) return false;

    // No AudioCapture — caller feeds the ring via feed_audio().  Log so it's
    // clear in headless runs that the mic was intentionally skipped.
    sched_log("[Scheduler] start_no_capture (headless; feed via feed_audio)");

    m_recording = true;
    m_windowerThread = std::thread(&TranscriptionScheduler::windower_loop, this);
    m_workerThreads.clear();
    m_workerThreads.reserve(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i)
        m_workerThreads.emplace_back(&TranscriptionScheduler::worker_loop, this, i);
    m_mergerThread = std::thread(&TranscriptionScheduler::merger_loop, this);
    return true;
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

    // Join all threads before tearing down capture/engines so none touch
    // `this` / m_engines after destruction.  In-flight transcribe_window
    // aborts in ms via the abort flag.
    join_thread(m_windowerThread);
    for (auto& t : m_workerThreads) join_thread(t);
    join_thread(m_mergerThread);
    m_workerThreads.clear();

    if (m_capture) {
        m_capture->stop();
        m_capture.reset();
    }
    m_engines.clear();
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
    auto& engine = m_engines[workerId];

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

        auto r = engine->transcribe_window(job.pcm.data(), (int)job.pcm.size(),
                                           &m_stopping);
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
                m_charsEmitted.fetch_add((long)text.size());
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
    s.total_chars        = m_charsEmitted.load();
    s.last_stop_ms       = m_lastStopMs.load();
    {
        std::lock_guard<std::mutex> lk(m_mergerMutex);
        s.next_emit = m_nextEmit;
        s.pending   = (int)m_pending.size();
    }
    return s;
}
