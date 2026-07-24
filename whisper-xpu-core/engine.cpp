#include "engine.h"

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
// Minimal WAV file reader (16-bit PCM, converts to float32)
// ---------------------------------------------------------------------------

static bool load_wav(const std::string& path, std::vector<float>& out, int expected_sr = 16000) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "[whisper-xpu] Failed to open: %s\n", path.c_str());
        return false;
    }

    auto read16 = [&]() -> uint16_t {
        uint16_t v;
        file.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    auto read32 = [&]() -> uint32_t {
        uint32_t v;
        file.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };

    // RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "[whisper-xpu] Not a RIFF file: %s\n", path.c_str());
        return false;
    }
    read32(); // file size
    char wave[4];
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[whisper-xpu] Not a WAVE file: %s\n", path.c_str());
        return false;
    }

    int channels = 0;
    int sample_rate = 0;
    int bits_per_sample = 0;
    std::vector<int16_t> pcm16;

    // Read chunks until 'data' is found
    while (file.good()) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        uint32_t chunk_size = read32();

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = read16();
            channels = read16();
            sample_rate = static_cast<int>(read32());
            read32(); // byte rate
            read16(); // block align
            bits_per_sample = read16();
            if (audio_format != 1) { // 1 = PCM
                fprintf(stderr, "[whisper-xpu] Unsupported WAV format (%d), need PCM\n", audio_format);
                return false;
            }
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            size_t data_bytes = chunk_size;
            if (bits_per_sample == 16) {
                size_t n_samples = data_bytes / 2;
                pcm16.resize(n_samples);
                file.read(reinterpret_cast<char*>(pcm16.data()), static_cast<std::streamsize>(data_bytes));
            } else if (bits_per_sample == 32) {
                size_t n_samples = data_bytes / 4;
                pcm16.resize(n_samples);
                for (size_t i = 0; i < n_samples && file.good(); ++i) {
                    int32_t s32;
                    file.read(reinterpret_cast<char*>(&s32), 4);
                    pcm16[i] = static_cast<int16_t>(s32 >> 16);
                }
            } else if (bits_per_sample == 8) {
                size_t n_samples = data_bytes;
                pcm16.resize(n_samples);
                for (size_t i = 0; i < n_samples && file.good(); ++i) {
                    uint8_t s8;
                    file.read(reinterpret_cast<char*>(&s8), 1);
                    pcm16[i] = static_cast<int16_t>((static_cast<int>(s8) - 128) << 8);
                }
            } else {
                fprintf(stderr, "[whisper-xpu] Unsupported bits per sample: %d\n", bits_per_sample);
                return false;
            }
        } else {
            // Skip unknown chunks
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (pcm16.empty() || channels == 0 || sample_rate == 0) {
        fprintf(stderr, "[whisper-xpu] Invalid WAV file: no data\n");
        return false;
    }

    // Convert to mono float32 at expected_sr
    // Step 1: mix down to mono
    std::vector<float> mono_float;
    if (channels == 1) {
        mono_float.resize(pcm16.size());
        for (size_t i = 0; i < pcm16.size(); ++i) {
            mono_float[i] = pcm16[i] / 32768.0f;
        }
    } else {
        size_t n_frames = pcm16.size() / channels;
        mono_float.resize(n_frames);
        for (size_t i = 0; i < n_frames; ++i) {
            float sum = 0;
            for (int c = 0; c < channels; ++c) {
                sum += pcm16[i * channels + c] / 32768.0f;
            }
            mono_float[i] = sum / channels;
        }
    }

    // Step 2: resample if needed (simple linear interpolation)
    if (sample_rate != expected_sr) {
        double ratio = static_cast<double>(expected_sr) / sample_rate;
        size_t new_len = static_cast<size_t>(mono_float.size() * ratio);
        std::vector<float> resampled(new_len);
        for (size_t i = 0; i < new_len; ++i) {
            double src_pos = i / ratio;
            size_t src_i = static_cast<size_t>(src_pos);
            double frac = src_pos - src_i;
            float s0 = mono_float[std::min(src_i, mono_float.size() - 1)];
            float s1 = mono_float[std::min(src_i + 1, mono_float.size() - 1)];
            resampled[i] = s0 + static_cast<float>(frac) * (s1 - s0);
        }
        out.swap(resampled);
    } else {
        out.swap(mono_float);
    }

    fprintf(stderr, "[whisper-xpu] Loaded WAV: %s (%d ch, %d Hz, %zu samples, %d-bit)\n",
            path.c_str(), channels, sample_rate, out.size(), bits_per_sample);
    return true;
}

// ---------------------------------------------------------------------------
// PIMPL structure
// ---------------------------------------------------------------------------

struct Engine::Impl {
    struct whisper_context* ctx = nullptr;
    bool gpu_initialized = false;
    std::string device_desc;
    int  n_threads = 0;

    Impl() {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads < 1) n_threads = 4;
    }

    ~Impl() {
        if (ctx) {
            whisper_free(ctx);
            ctx = nullptr;
        }
    }

    // Probe for SYCL device via ggml-sycl backend API
    bool probe_sycl_device() {
#ifdef WHISPER_XPU_HAS_SYCL
        int device_count = ggml_backend_sycl_get_device_count();
        if (device_count < 1) {
            fprintf(stderr, "[whisper-xpu] No SYCL devices found (%d)\n", device_count);
            return false;
        }

        char desc[256] = {0};
        ggml_backend_sycl_get_device_description(0, desc, sizeof(desc));

        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(0, &free_mem, &total_mem);

        std::ostringstream oss;
        oss << desc;
        if (total_mem > 0) {
            oss << " | VRAM: " << (total_mem / (1024 * 1024)) << " MB";
        }
        device_desc = oss.str();
        fprintf(stderr, "[whisper-xpu] SYCL device: %s\n", device_desc.c_str());
        return true;
#else
        (void)0;
        return false;
#endif
    }
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Engine::Engine(const std::string& model_path, bool use_gpu)
    : pimpl_(std::make_unique<Impl>())
{
    if (use_gpu) {
#ifdef WHISPER_XPU_HAS_SYCL
        pimpl_->gpu_initialized = pimpl_->probe_sycl_device();
        if (!pimpl_->gpu_initialized) {
            fprintf(stderr, "[whisper-xpu] GPU init failed, falling back to CPU\n");
        }
#else
        fprintf(stderr, "[whisper-xpu] Not compiled with SYCL support, using CPU\n");
#endif
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = pimpl_->gpu_initialized;

    pimpl_->ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!pimpl_->ctx) {
        throw std::runtime_error("Failed to load whisper model: " + model_path);
    }

    if (!pimpl_->gpu_initialized) {
        pimpl_->device_desc = "CPU (" + std::to_string(pimpl_->n_threads) + " threads)";
    }

    fprintf(stderr, "[whisper-xpu] Model loaded: %s\n", model_path.c_str());
    fprintf(stderr, "[whisper-xpu] Device: %s\n", device_description().c_str());
}

Engine::~Engine() = default;

// ---------------------------------------------------------------------------
// transcribe_file
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_file(const std::string& audio_path) {
    if (!pimpl_->ctx) {
        throw std::runtime_error("Engine not initialized");
    }

    // Load WAV file
    std::vector<float> pcm;
    if (!load_wav(audio_path, pcm, 16000)) {
        return TranscriptionResult{"", 0, 0, pimpl_->gpu_initialized};
    }
    if (pcm.empty()) {
        return TranscriptionResult{"", 0, 0, pimpl_->gpu_initialized};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime     = false;
    wparams.print_progress     = false;
    wparams.print_timestamps   = false;
    wparams.print_special      = false;
    wparams.translate          = false;
    wparams.language           = "auto";
    wparams.detect_language    = true;
    wparams.n_threads          = pimpl_->n_threads;
    wparams.offset_ms          = 0;
    wparams.duration_ms        = 0;
    wparams.single_segment     = false;

    if (whisper_full(pimpl_->ctx, wparams, pcm.data(), static_cast<int>(pcm.size())) != 0) {
        throw std::runtime_error("whisper_full() failed for: " + audio_path);
    }

    std::string result;
    const int n_segments = whisper_full_n_segments(pimpl_->ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(pimpl_->ctx, i);
        if (text) {
            if (i > 0) result += " ";
            result += text;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return TranscriptionResult{
        result,
        duration_ms,
        n_segments,
        pimpl_->gpu_initialized
    };
}

// ---------------------------------------------------------------------------
// transcribe_stream
// ---------------------------------------------------------------------------

TranscriptionResult Engine::transcribe_stream(AudioSampleCallback callback) {
    if (!pimpl_->ctx) {
        throw std::runtime_error("Engine not initialized");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr int SAMPLE_RATE = 16000;
    constexpr int CHUNK_MS    = 3000;
    constexpr int CHUNK_SIZE  = SAMPLE_RATE * CHUNK_MS / 1000;

    std::vector<float> pcm_buffer(CHUNK_SIZE);
    std::string full_result;
    int total_segments = 0;

    while (true) {
        size_t samples_read = callback(pcm_buffer.data(), CHUNK_SIZE);
        if (samples_read == 0) break;

        pcm_buffer.resize(samples_read);

        auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime     = false;
        wparams.print_progress     = false;
        wparams.print_timestamps   = false;
        wparams.print_special      = false;
        wparams.translate          = false;
        wparams.language           = "auto";
        wparams.detect_language    = true;
        wparams.n_threads          = pimpl_->n_threads;
        wparams.single_segment     = true;
        wparams.no_context         = true;
        wparams.no_timestamps      = true;

        int ret = whisper_full(pimpl_->ctx, wparams, pcm_buffer.data(),
                               static_cast<int>(samples_read));
        if (ret != 0) {
            fprintf(stderr, "[whisper-xpu] Warning: whisper_full() failed on chunk\n");
            continue;
        }

        const int n = whisper_full_n_segments(pimpl_->ctx);
        for (int i = 0; i < n; ++i) {
            const char* text = whisper_full_get_segment_text(pimpl_->ctx, i);
            if (text && strlen(text) > 0) {
                full_result += text;
                full_result += " ";
            }
        }
        total_segments += n;

        pcm_buffer.resize(CHUNK_SIZE);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (!full_result.empty() && full_result.back() == ' ') {
        full_result.pop_back();
    }

    return TranscriptionResult{
        full_result,
        duration_ms,
        total_segments,
        pimpl_->gpu_initialized
    };
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool Engine::is_gpu_enabled() const {
    return pimpl_->gpu_initialized;
}

std::string Engine::device_description() const {
    return pimpl_->device_desc;
}

} // namespace whisper_xpu
