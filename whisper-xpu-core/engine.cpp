#include "merge_segments.h"
#include "engine.h"
#include "device_detect.h"

#include "whisper.h"
#include "ggml-sycl.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>

namespace {
// Abort callbacks as plain free functions (cdecl), not lambdas.  Under icpx
// on Windows, converting a lambda to a function pointer via unary + decorates
// the pointer with __attribute__((vectorcall)), which is incompatible with
// ggml_abort_callback / whisper_encoder_begin_callback (cdecl) and fails to
// compile.  Free functions keep cdecl and assign cleanly.  Logic is identical
// to the lambdas they replace.
static bool xpu_abort_cb(void* ud) {
    return static_cast<const std::atomic<bool>*>(ud)->load();
}
static bool xpu_encoder_begin_cb(struct whisper_context*,
                                 struct whisper_state*, void* ud) {
    return !static_cast<const std::atomic<bool>*>(ud)->load();
}
} // namespace

namespace whisper_xpu {

// ---------------------------------------------------------------------------
// WAV loader
// ---------------------------------------------------------------------------

static bool load_wav(const std::string& path, std::vector<float>& out, int expected_sr = 16000) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    auto r16 = [&]() -> uint16_t { uint16_t v; file.read((char*)&v, sizeof(v)); return v; };
    auto r32 = [&]() -> uint32_t { uint32_t v; file.read((char*)&v, sizeof(v)); return v; };

    char riff[4]; file.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;
    r32();
    char wave[4]; file.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    int ch = 0, sr = 0, bps = 0;
    std::vector<int16_t> pcm;
    while (file.good()) {
        char cid[4]; file.read(cid, 4);
        uint32_t csz = r32();
        if (memcmp(cid, "fmt ", 4) == 0) {
            r16(); ch = r16(); sr = (int)r32(); r32(); r16(); bps = r16();
        } else if (memcmp(cid, "data", 4) == 0) {
            if (bps == 16) { pcm.resize(csz/2); file.read((char*)pcm.data(), csz); }
            else file.seekg(csz, std::ios::cur);
        } else file.seekg(csz, std::ios::cur);
    }
    if (pcm.empty() || ch == 0) return false;

    std::vector<float> mono;
    if (ch == 1) {
        mono.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); i++) mono[i] = pcm[i] / 32768.0f;
    } else {
        size_t nf = pcm.size() / ch;
        mono.resize(nf);
        for (size_t i = 0; i < nf; i++) {
            float s = 0; for (int c = 0; c < ch; c++) s += pcm[i*ch+c] / 32768.0f;
            mono[i] = s / ch;
        }
    }

    if (sr != expected_sr) {
        double ratio = (double)expected_sr / sr;
        size_t nl = (size_t)(mono.size() * ratio);
        std::vector<float> r(nl);
        for (size_t i = 0; i < nl; i++) {
            double pos = i / ratio; size_t si = (size_t)pos; double fr = pos - si;
            float s0 = mono[std::min(si, mono.size()-1)];
            float s1 = mono[std::min(si+1, mono.size()-1)];
            r[i] = s0 + (float)fr * (s1 - s0);
        }
        out.swap(r);
    } else out.swap(mono);
    return true;
}

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------

struct Engine::Impl {
    struct whisper_context* ctx = nullptr;
    bool gpu_initialized = false;
    std::string device_desc;
    int device_id = kDeviceCPU;
    int n_threads = 0;
    // Language detected on the first streaming chunk and pinned for all
    // subsequent chunks.  whisper pads every chunk to a 30 s window and
    // language="auto" runs a SEPARATE encoder pass for detection — so
    // re-detecting per chunk roughly doubles the per-chunk cost on CPU.
    // Detecting once halves it and is the right design for a single
    // speaker/session.  Empty ⇒ not yet detected (use "auto").
    std::string detected_language;

    Impl() { n_threads = (int)std::thread::hardware_concurrency(); if (n_threads<1) n_threads=4; }
    ~Impl() { if (ctx) { whisper_free(ctx); ctx=nullptr; } }

    bool probe_sycl_device(int dev_id) {
        // ggml_backend_sycl_* are imported from whisper_xpu_sycl_core.dll.
        // The icpx-compiled DLL handles its own failures internally.
        int n = ggml_backend_sycl_get_device_count();
        if (n < 1 || dev_id >= n) return false;
        char desc[256]={0};
        ggml_backend_sycl_get_device_description(dev_id, desc, sizeof(desc));
        size_t fm=0, tm=0; ggml_backend_sycl_get_device_memory(dev_id, &fm, &tm);
        std::ostringstream os; os << desc;
        if (tm > 0) { os << " | VRAM: " << (tm/(1024*1024)) << " MB"; if (fm>0) os << " (free: " << (fm/(1024*1024)) << " MB)"; }
        device_desc = os.str();
        fprintf(stderr, "[whisper-xpu] SYCL device %d: %s\n", dev_id, device_desc.c_str());
        return true;
    }
};

Engine::Engine(const std::string& path, int device_id, int n_threads) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->device_id = device_id;
    // 0 (default) ⇒ hardware_concurrency (set in Impl ctor).  A pool caller
    // passes core_count/pool_size so the total across workers ≈ core count.
    if (n_threads > 0) pimpl_->n_threads = n_threads;
    if (device_id >= 0) {
        pimpl_->gpu_initialized = pimpl_->probe_sycl_device(device_id);
        if (!pimpl_->gpu_initialized) fprintf(stderr, "[whisper-xpu] GPU init failed\n");
    }
    auto cp = whisper_context_default_params();
    cp.use_gpu = pimpl_->gpu_initialized; cp.gpu_device = device_id;
    cp.flash_attn = pimpl_->gpu_initialized;  // flash_attn only on GPU
    pimpl_->ctx = whisper_init_from_file_with_params(path.c_str(), cp);
    if (!pimpl_->ctx) throw std::runtime_error("Failed to load: " + path);
    if (!pimpl_->gpu_initialized) pimpl_->device_desc = "CPU (" + std::to_string(pimpl_->n_threads) + "t)";
}

Engine::~Engine() = default;

// ---------------------------------------------------------------------------
// transcribe_file
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_file(const std::string& audio_path, const VadConfig& vad) {
    if (!pimpl_->ctx) throw std::runtime_error("no ctx");
    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm, WHISPER_SAMPLE_RATE) || pcm.empty())
        return TranscriptionResult{};

    auto t0 = std::chrono::high_resolution_clock::now();
    auto wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_realtime=false; wp.print_progress=false; wp.print_timestamps=false;
    wp.print_special=false; wp.translate=false; wp.language="auto";
    // NOTE: detect_language must be false here.  In this whisper.cpp build
    // detect_language=true means "detect language then return immediately
    // without transcribing" (whisper.cpp:6845), which was the root cause of
    // the 0-char / 0-segment output.  language="auto" still auto-detects and
    // then proceeds to transcribe.
    wp.detect_language=false; wp.n_threads=pimpl_->n_threads;

    if (vad.enabled) {
        wp.vad_model_path = vad.vad_model_path;
        wp.vad = true;
        wp.vad_params.threshold            = vad.vad_threshold;
        wp.vad_params.min_speech_duration_ms = vad.min_speech_duration_ms;
        wp.vad_params.min_silence_duration_ms = vad.min_silence_duration_ms;
        wp.vad_params.max_speech_duration_s  = vad.max_speech_duration_s;
        wp.vad_params.speech_pad_ms          = vad.speech_pad_ms;
    }

    if (whisper_full(pimpl_->ctx, wp, pcm.data(), (int)pcm.size()) != 0)
        throw std::runtime_error("whisper_full failed: " + audio_path);

    int ns = whisper_full_n_segments(pimpl_->ctx);
    std::vector<std::string> segs(ns);
    for (int i = 0; i < ns; i++) {
        const char* t = whisper_full_get_segment_text(pimpl_->ctx, i);
        segs[i] = t ? t : "";
    }
    std::string merged = merge_segments(segs);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return TranscriptionResult{merged, ms, ns, pimpl_->gpu_initialized};
}

// ---------------------------------------------------------------------------
// transcribe_chunk  (single PCM block, abortable)
// ---------------------------------------------------------------------------

std::string Engine::transcribe_chunk(const float* pcm, int n_samples,
                                     const std::atomic<bool>* abort_flag) {
    if (!pimpl_->ctx || n_samples <= 0) return {};

    auto wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_realtime=false; wp.print_progress=false; wp.print_timestamps=false;
    wp.print_special=false; wp.translate=false;
    // detect_language=false: see transcribe_file — true means detect-and-exit.
    wp.detect_language=false; wp.n_threads=pimpl_->n_threads;
    wp.single_segment=true; wp.no_context=true; wp.no_timestamps=true;

    // Language handling: on the first chunk use "auto" (whisper detects via
    // its own encoder pass) and cache the result; on subsequent chunks pin
    // the cached language so the detection pass is skipped — halving the
    // per-chunk encoder cost on CPU (critical for real-time chunking).
    const bool first_chunk = pimpl_->detected_language.empty();
    wp.language = first_chunk ? "auto" : pimpl_->detected_language.c_str();

    // Wire whisper's abort hooks to the caller's flag so an in-flight chunk
    // bails in milliseconds when stop() is requested.  abort_callback is
    // polled before each ggml compute op; encoder_begin_callback gates the
    // (expensive) encoder.  Both read the same atomic<bool>.
    if (abort_flag) {
        wp.abort_callback = xpu_abort_cb;
        wp.abort_callback_user_data = const_cast<void*>(
            static_cast<const void*>(abort_flag));
        // encoder_begin returns false ⇒ abort (skip encoder).
        wp.encoder_begin_callback = xpu_encoder_begin_cb;
        wp.encoder_begin_callback_user_data = const_cast<void*>(
            static_cast<const void*>(abort_flag));
    }

    // whisper_full returns non-zero on abort or failure → treat as no text.
    if (whisper_full(pimpl_->ctx, wp, pcm, n_samples) != 0) return {};

    // Cache the detected language from the first chunk so subsequent chunks
    // pin it and skip the detection encoder pass.
    if (first_chunk) {
        int lid = whisper_full_lang_id(pimpl_->ctx);
        if (lid >= 0) pimpl_->detected_language = whisper_lang_str(lid);
    }

    int ns = whisper_full_n_segments(pimpl_->ctx);
    std::string text;
    for (int i = 0; i < ns; i++) {
        const char* t = whisper_full_get_segment_text(pimpl_->ctx, i);
        if (t && *t) { text += t; text += " "; }
    }
    if (!text.empty() && text.back() == ' ') text.pop_back();
    return text;
}

// ---------------------------------------------------------------------------
// Parallel-pool: shared-context state + transcribe_window_with_state
//
// The scheduler's 4 workers share ONE Engine (one whisper_context = one
// read-only model copy).  Each worker owns a whisper_state (KV cache,
// logits, decoders — all per-state, allocated in whisper_init_state) and
// runs whisper_full_with_state(ctx, st, …) concurrently.  This is whisper.cpp's
// "one context + N states" pattern (issue #523): pool RAM stays at ~1× model
// + 4 small states instead of 4× model.  (The now-removed transcribe_window
// used ctx->state — only one such state per context, so it could not run
// concurrently across workers.)
// ---------------------------------------------------------------------------

whisper_state* Engine::create_state() {
    if (!pimpl_->ctx) return nullptr;
    return whisper_init_state(pimpl_->ctx);
}

void Engine::free_state(whisper_state* st) {
    if (st) whisper_free_state(st);
}

ChunkResult Engine::transcribe_window_with_state(whisper_state* st, int n_threads,
                                                 std::string& detected_language,
                                                 const float* pcm, int n_samples,
                                                 const std::atomic<bool>* abort_flag) {
    ChunkResult result;
    if (!pimpl_->ctx || !st || n_samples <= 0) return result;

    auto wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_realtime=false; wp.print_progress=false; wp.print_timestamps=false;
    wp.print_special=false; wp.translate=false;
    // detect_language=false: see transcribe_file — true means detect-and-exit.
    wp.detect_language=false;
    wp.n_threads = n_threads > 0 ? n_threads : pimpl_->n_threads;
    // Multi-segment + timestamps: REQUIRED for overlap merging (the merger
    // keys on each segment's midpoint).
    wp.single_segment=false; wp.no_timestamps=false; wp.no_context=true;

    // Per-worker language cache (caller-owned): first window auto-detects and
    // pins; later windows skip the detection encoder pass.
    const bool first_chunk = detected_language.empty();
    wp.language = first_chunk ? "auto" : detected_language.c_str();

    // Abort wiring — identical to transcribe_chunk: abort_callback polled
    // before each ggml compute op; encoder_begin gates the expensive encoder.
    // The callbacks receive (ctx, state, ud) but ignore ctx/state and read ud.
    if (abort_flag) {
        wp.abort_callback = xpu_abort_cb;
        wp.abort_callback_user_data = const_cast<void*>(
            static_cast<const void*>(abort_flag));
        wp.encoder_begin_callback = xpu_encoder_begin_cb;
        wp.encoder_begin_callback_user_data = const_cast<void*>(
            static_cast<const void*>(abort_flag));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = whisper_full_with_state(pimpl_->ctx, st, wp, pcm, n_samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // whisper_full_with_state returns non-zero on abort OR hard failure → no
    // usable segments either way.  Flag aborted only when the caller's flag
    // was actually set (distinguishes a stop() abort from a real failure).
    if (rc != 0) {
        result.aborted = abort_flag && abort_flag->load();
        return result;
    }

    // Cache the detected language so subsequent windows pin it and skip the
    // detection encoder pass.
    if (first_chunk) {
        int lid = whisper_full_lang_id_from_state(st);
        if (lid >= 0) detected_language = whisper_lang_str(lid);
    }

    // Extract timestamped segments.  _from_state accessors read from `st`.
    // t0/t1 are 10-millisecond units (centiseconds) → ×10 for ms.
    int ns = whisper_full_n_segments_from_state(st);
    result.segments.reserve(ns);
    for (int i = 0; i < ns; i++) {
        ChunkSegment seg;
        const char* t = whisper_full_get_segment_text_from_state(st, i);
        seg.text  = t ? t : "";
        seg.t0_ms = whisper_full_get_segment_t0_from_state(st, i) * 10;
        seg.t1_ms = whisper_full_get_segment_t1_from_state(st, i) * 10;
        result.segments.push_back(std::move(seg));
    }
    return result;
}

// ---------------------------------------------------------------------------
// transcribe_stream
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_stream(AudioSampleCallback cb,
                                              const std::atomic<bool>* abort_flag) {
    if (!pimpl_->ctx) throw std::runtime_error("no ctx");
    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr int SR=WHISPER_SAMPLE_RATE, CS=SR*3;
    std::vector<float> buf(CS);
    std::string full; int chunks=0;

    while (true) {
        // Fast exit before pulling if already aborted.
        if (abort_flag && abort_flag->load()) break;
        size_t n = cb(buf.data(), CS); if (n==0) break;

        std::string chunk_text = transcribe_chunk(buf.data(), (int)n, abort_flag);
        if (!chunk_text.empty()) { full += chunk_text; full += " "; ++chunks; }
        buf.resize(CS);
    }
    if (!full.empty() && full.back()==' ') full.pop_back();
    auto t1 = std::chrono::high_resolution_clock::now();
    return TranscriptionResult{full, (double)std::chrono::duration<double,std::milli>(t1-t0).count(), chunks, pimpl_->gpu_initialized};
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

BenchmarkResult Engine::benchmark(const std::string& path, const VadConfig& vad) {
    std::vector<float> pcm;
    double audio_s = 0;
    if (load_wav(path, pcm, WHISPER_SAMPLE_RATE)) audio_s = (double)pcm.size() / WHISPER_SAMPLE_RATE;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r = transcribe_file(path, vad);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    BenchmarkResult br{audio_s, ms, ms/(audio_s*1000.0), ms/(audio_s*1000.0)};
    fprintf(stderr,"\n[bench] audio=%.1fs proc=%.0fms RTF=%.2f segs=%d vad=%s\n",
            audio_s, ms, br.rtf, r.segment_count, vad.enabled?"yes":"no");
    return br;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool Engine::is_gpu_enabled() const { return pimpl_->gpu_initialized; }
std::string Engine::device_description() const { return pimpl_->device_desc; }
int Engine::device_id() const { return pimpl_->device_id; }

} // namespace whisper_xpu
