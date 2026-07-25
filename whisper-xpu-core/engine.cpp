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

    auto read16 = [&]() -> uint16_t { uint16_t v; file.read((char*)&v, sizeof(v)); return v; };
    auto read32 = [&]() -> uint32_t { uint32_t v; file.read((char*)&v, sizeof(v)); return v; };

    char riff[4]; file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) return false;
    read32();
    char wave[4]; file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) return false;

    int channels = 0, sample_rate = 0, bits_per_sample = 0;
    std::vector<int16_t> pcm16;

    while (file.good()) {
        char cid[4]; file.read(cid, 4);
        uint32_t csz = read32();
        if (memcmp(cid, "fmt ", 4) == 0) {
            read16(); channels = read16(); sample_rate = (int)read32();
            read32(); read16(); bits_per_sample = read16();
            if (channels == 0) return false;
        } else if (memcmp(cid, "data", 4) == 0) {
            if (bits_per_sample == 16) { pcm16.resize(csz / 2); file.read((char*)pcm16.data(), csz); }
            else file.seekg(csz, std::ios::cur);
        } else file.seekg(csz, std::ios::cur);
    }
    if (pcm16.empty() || sample_rate == 0) return false;

    // mono f32
    std::vector<float> mono;
    if (channels == 1) {
        mono.resize(pcm16.size());
        for (size_t i = 0; i < pcm16.size(); i++) mono[i] = pcm16[i] / 32768.0f;
    } else {
        size_t nf = pcm16.size() / channels;
        mono.resize(nf);
        for (size_t i = 0; i < nf; i++) {
            float s = 0; for (int c = 0; c < channels; c++) s += pcm16[i*channels+c] / 32768.0f;
            mono[i] = s / channels;
        }
    }

    if (sample_rate != expected_sr) {
        double ratio = (double)expected_sr / sample_rate;
        size_t nl = (size_t)(mono.size() * ratio);
        std::vector<float> r(nl);
        for (size_t i = 0; i < nl; i++) {
            double pos = i / ratio;
            size_t si = (size_t)pos; double fr = pos - si;
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

    Impl() {
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 4;
    }
    ~Impl() { if (ctx) { whisper_free(ctx); ctx = nullptr; } }

    bool probe_sycl_device(int dev_id) {
#ifdef WHISPER_XPU_HAS_SYCL
        int n = ggml_backend_sycl_get_device_count();
        if (n < 1) return false;
        if (dev_id >= n) return false;
        char desc[256] = {0};
        ggml_backend_sycl_get_device_description(dev_id, desc, sizeof(desc));
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(dev_id, &free_mem, &total_mem);
        std::ostringstream oss; oss << desc;
        if (total_mem > 0) {
            oss << " | VRAM: " << (total_mem/(1024*1024)) << " MB";
            if (free_mem > 0) oss << " (free: " << (free_mem/(1024*1024)) << " MB)";
        }
        device_desc = oss.str();
        fprintf(stderr, "[whisper-xpu] SYCL device %d: %s\n", dev_id, device_desc.c_str());
        return true;
#else
        (void)dev_id; return false;
#endif
    }
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Engine::Engine(const std::string& model_path, int device_id)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->device_id = device_id;
    if (device_id >= 0) {
#ifdef WHISPER_XPU_HAS_SYCL
        pimpl_->gpu_initialized = pimpl_->probe_sycl_device(device_id);
        if (!pimpl_->gpu_initialized) fprintf(stderr, "[whisper-xpu] GPU init failed, CPU fallback\n");
#else
        fprintf(stderr, "[whisper-xpu] No SYCL support, using CPU\n");
#endif
    }
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = pimpl_->gpu_initialized;
    cparams.gpu_device = device_id;
    pimpl_->ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!pimpl_->ctx) throw std::runtime_error("Failed to load model: " + model_path);
    if (!pimpl_->gpu_initialized) pimpl_->device_desc = "CPU (" + std::to_string(pimpl_->n_threads) + " threads)";
    fprintf(stderr, "[whisper-xpu] Model loaded, device: %s\n", device_description().c_str());
}

Engine::~Engine() = default;

// ---------------------------------------------------------------------------
// transcribe_file (with optional VAD)
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_file(const std::string& audio_path, const VadConfig& vad) {
    if (!pimpl_->ctx) throw std::runtime_error("Engine not initialized");

    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm, WHISPER_SAMPLE_RATE))
        return TranscriptionResult{"", 0, 0, pimpl_->gpu_initialized};
    if (pcm.empty())
        return TranscriptionResult{"", 0, 0, pimpl_->gpu_initialized};

    int n_samples = (int)pcm.size();
    auto start_time = std::chrono::high_resolution_clock::now();

    auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = "auto";
    wparams.detect_language  = true;
    wparams.n_threads        = pimpl_->n_threads;
    wparams.offset_ms        = 0;
    wparams.duration_ms      = 0;
    wparams.single_segment   = false;

    if (vad.enabled) {
        wparams.vad = true;
        wparams.vad_params.threshold            = vad.vad_threshold;
        wparams.vad_params.min_speech_duration_ms = vad.min_speech_duration_ms;
        wparams.vad_params.min_silence_duration_ms = vad.min_silence_duration_ms;
        wparams.vad_params.max_speech_duration_s  = vad.max_speech_duration_s;
        wparams.vad_params.speech_pad_ms          = vad.speech_pad_ms;
    }

    if (whisper_full(pimpl_->ctx, wparams, pcm.data(), n_samples) != 0)
        throw std::runtime_error("whisper_full() failed for: " + audio_path);

    std::string result;
    int n_seg = whisper_full_n_segments(pimpl_->ctx);
    for (int i = 0; i < n_seg; i++) {
        const char* text = whisper_full_get_segment_text(pimpl_->ctx, i);
        if (text) {
            // Dedup: skip if this segment's text is a near-duplicate of the last appended
            if (i > 0) {
                std::string prev = result;
                // Check if the new text is mostly contained in the last part of result
                // Simple heuristic: if result already ends with text, skip
                if (prev.size() >= strlen(text) &&
                    prev.compare(prev.size() - strlen(text), strlen(text), text) == 0)
                    continue;
            }
            if (!result.empty()) result += " ";
            result += text;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
    return TranscriptionResult{result, ms, n_seg, pimpl_->gpu_initialized};
}

// ---------------------------------------------------------------------------
// transcribe_stream
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_stream(AudioSampleCallback callback) {
    if (!pimpl_->ctx) throw std::runtime_error("Engine not initialized");

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr int SR = WHISPER_SAMPLE_RATE;
    constexpr int CHUNK_MS = 3000;
    constexpr int CHUNK_SIZE = SR * CHUNK_MS / 1000;

    std::vector<float> buf(CHUNK_SIZE);
    std::string full;
    int total_seg = 0;

    while (true) {
        size_t n = callback(buf.data(), CHUNK_SIZE);
        if (n == 0) break;
        buf.resize(n);

        auto wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wp.print_realtime = false; wp.print_progress = false;
        wp.print_timestamps = false; wp.print_special = false;
        wp.translate = false; wp.language = "auto";
        wp.detect_language = true; wp.n_threads = pimpl_->n_threads;
        wp.single_segment = true; wp.no_context = true; wp.no_timestamps = true;

        if (whisper_full(pimpl_->ctx, wp, buf.data(), (int)n) != 0) continue;

        int ns = whisper_full_n_segments(pimpl_->ctx);
        for (int i = 0; i < ns; i++) {
            const char* t = whisper_full_get_segment_text(pimpl_->ctx, i);
            if (t && strlen(t) > 0) { full += t; full += " "; }
        }
        total_seg += ns;
        buf.resize(CHUNK_SIZE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
    if (!full.empty() && full.back() == ' ') full.pop_back();
    return TranscriptionResult{full, ms, total_seg, pimpl_->gpu_initialized};
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

BenchmarkResult Engine::benchmark(const std::string& audio_path, const VadConfig& vad) {
    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "[whisper-xpu] benchmark: failed to load %s\n", audio_path.c_str());
        return BenchmarkResult{};
    }

    double audio_duration_s = (double)pcm.size() / WHISPER_SAMPLE_RATE;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = transcribe_file(audio_path, vad);
    auto t1 = std::chrono::high_resolution_clock::now();

    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    BenchmarkResult br;
    br.total_audio_duration_s = audio_duration_s;
    br.processing_time_ms     = wall_ms;
    br.realtime_factor        = wall_ms / (audio_duration_s * 1000.0);
    br.rtf                    = br.realtime_factor;

    fprintf(stderr, "\n[whisper-xpu] === Benchmark ===\n");
    fprintf(stderr, "  Audio duration:  %.1f s\n", audio_duration_s);
    fprintf(stderr, "  Processing time: %.0f ms (%.2f s)\n", wall_ms, wall_ms / 1000.0);
    fprintf(stderr, "  RTF:             %.2f (%.1fx realtime)\n", br.rtf, 1.0 / br.rtf);
    fprintf(stderr, "  Segments:        %d\n", result.segment_count);
    fprintf(stderr, "  VAD:             %s\n", vad.enabled ? "yes" : "no");
    fprintf(stderr, "  Output length:   %zu chars\n", result.text.size());
    fprintf(stderr, "==================\n");

    return br;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool Engine::is_gpu_enabled() const { return pimpl_->gpu_initialized; }
std::string Engine::device_description() const { return pimpl_->device_desc; }
int Engine::device_id() const { return pimpl_->device_id; }

} // namespace whisper_xpu
