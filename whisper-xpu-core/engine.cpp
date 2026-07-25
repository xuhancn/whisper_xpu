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
// merge_segments  (deduplicate overlapping suffixes)
// ---------------------------------------------------------------------------

// moved to merge_segments.cpp
std::string merge_segments(const std::vector<const char*>& segments) {
    std::string result;
    for (size_t i = 0; i < segments.size(); i++) {
        if (!segments[i]) continue;
        std::string cur = segments[i];
        if (cur.empty()) continue;

        if (i > 0 && !result.empty()) {
            size_t rlen = result.size(), clen = cur.size();

            // exact suffix duplicate
            if (clen <= rlen && result.compare(rlen - clen, clen, cur) == 0)
                continue;

            // partial overlap: trim matching portion from cur
            for (size_t o = std::min(rlen, clen); o > 4; o--) {
                if (result.compare(rlen - o, o, cur.c_str(), o) == 0) {
                    cur = cur.substr(o);
                    break;
                }
            }
            if (cur.empty()) continue;
        }
        if (!result.empty() && result.back() != ' ') result += ' ';
        result += cur;
    }
    return result;
}

// moved to merge_segments.cpp
std::string merge_segments(const std::vector<std::string>& segments) {
    std::vector<const char*> ptrs;
    ptrs.reserve(segments.size());
    for (const auto& s : segments) ptrs.push_back(s.c_str());
    return merge_segments(ptrs);
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

    Impl() { n_threads = (int)std::thread::hardware_concurrency(); if (n_threads<1) n_threads=4; }
    ~Impl() { if (ctx) { whisper_free(ctx); ctx=nullptr; } }

    bool probe_sycl_device(int dev_id) {
#ifdef WHISPER_XPU_HAS_SYCL
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
#else
        (void)dev_id; return false;
#endif
    }
};

Engine::Engine(const std::string& path, int device_id) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->device_id = device_id;
    if (device_id >= 0) {
#ifdef WHISPER_XPU_HAS_SYCL
        pimpl_->gpu_initialized = pimpl_->probe_sycl_device(device_id);
        if (!pimpl_->gpu_initialized) fprintf(stderr, "[whisper-xpu] GPU init failed\n");
#else
        fprintf(stderr, "[whisper-xpu] No SYCL\n");
#endif
    }
    auto cp = whisper_context_default_params();
    cp.use_gpu = pimpl_->gpu_initialized; cp.gpu_device = device_id;
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
    wp.detect_language=true; wp.n_threads=pimpl_->n_threads;

    if (vad.enabled) {
        wparams.vad_model_path = vad.vad_model_path;
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
// transcribe_stream
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_stream(AudioSampleCallback cb) {
    if (!pimpl_->ctx) throw std::runtime_error("no ctx");
    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr int SR=WHISPER_SAMPLE_RATE, CS=SR*3;
    std::vector<float> buf(CS);
    std::string full; int ts=0;

    while (true) {
        size_t n = cb(buf.data(), CS); if (n==0) break;
        buf.resize(n);
        auto wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wp.print_realtime=false; wp.print_progress=false; wp.print_timestamps=false;
        wp.print_special=false; wp.translate=false; wp.language="auto";
        wp.detect_language=true; wp.n_threads=pimpl_->n_threads;
        wp.single_segment=true; wp.no_context=true; wp.no_timestamps=true;
        if (whisper_full(pimpl_->ctx, wp, buf.data(), (int)n)!=0) continue;
        int ns = whisper_full_n_segments(pimpl_->ctx);
        for (int i=0;i<ns;i++) { const char*t=whisper_full_get_segment_text(pimpl_->ctx,i); if(t&&*t) {full+=t;full+=" ";}}
        ts+=ns; buf.resize(CS);
    }
    if (!full.empty() && full.back()==' ') full.pop_back();
    auto t1 = std::chrono::high_resolution_clock::now();
    return TranscriptionResult{full, (double)std::chrono::duration<double,std::milli>(t1-t0).count(), ts, pimpl_->gpu_initialized};
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
