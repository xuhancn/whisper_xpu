#include "engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Paths injected by CMake
#ifndef TEST_WAV_DIR
#define TEST_WAV_DIR "."
#endif
#ifndef VAD_MODEL_DIR
#define VAD_MODEL_DIR "."
#endif

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    std::string audio_path = std::string(TEST_WAV_DIR) + "/trump_60s_final.wav";
    int device_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model_path = argv[++i];
        else if (a == "--audio" && i+1 < argc) audio_path = argv[++i];
        else if (a == "--cpu") device_id = -1;
        else if (a == "--help") {
            printf("Usage: bench_vad --model <path> [--audio <wav>] [--cpu]\n");
            printf("Default audio: %s\n", audio_path.c_str());
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required\n");
        return 1;
    }

    fprintf(stderr, "Loading engine (device %d)...\n", device_id);
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Audio: %s\n", audio_path.c_str());
    fprintf(stderr, "VAD model: %s\n\n", std::string(VAD_MODEL_DIR) + "/ggml-vad.bin");

    whisper_xpu::Engine engine(model_path, device_id);
    fprintf(stderr, "Device: %s\n\n", engine.device_description().c_str());

    // ── Without VAD ──
    fprintf(stderr, "=== transcribe_file (no VAD) ===\n");
    auto r0 = engine.transcribe_file(audio_path);
    fprintf(stderr, "  chars=%zu segs=%d time=%.0fms\n",
            r0.text.size(), r0.segment_count, r0.processing_time_ms);

    // ── With VAD ──
    whisper_xpu::VadConfig vad;
    vad.enabled = true;
    vad.max_speech_duration_s = 5.0f;
    vad.speech_pad_ms = 500;
    vad.vad_threshold = 0.5f;
    vad.vad_model_path = (std::string(VAD_MODEL_DIR) + "/ggml-vad.bin").c_str();

    fprintf(stderr, "\n=== transcribe_file (VAD) ===\n");
    auto r1 = engine.transcribe_file(audio_path, vad);
    fprintf(stderr, "  chars=%zu segs=%d time=%.0fms\n",
            r1.text.size(), r1.segment_count, r1.processing_time_ms);

    // ── Benchmark (VAD) ──
    fprintf(stderr, "\n=== benchmark (VAD) ===\n");
    auto br = engine.benchmark(audio_path, vad);

    // ── Comparison ──
    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Non-VAD: %zu chars, %d segs, %.0f ms\n",
            r0.text.size(), r0.segment_count, r0.processing_time_ms);
    fprintf(stderr, "VAD:     %zu chars, %d segs, %.0f ms\n",
            r1.text.size(), r1.segment_count, r1.processing_time_ms);
    fprintf(stderr, "RTF:     %.2f (%.1fx realtime)\n", br.rtf, 1.0/br.rtf);

    // ── Sample output ──
    if (!r0.text.empty())
        fprintf(stderr, "\n=== Non-VAD (first 500 chars) ===\n%.*s\n", 500, r0.text.c_str());
    if (!r1.text.empty())
        fprintf(stderr, "\n=== VAD (first 500 chars) ===\n%.*s\n", 500, r1.text.c_str());

    // Basic sanity checks
    bool pass = true;
    if (r0.segment_count == 0 && r1.segment_count == 0) {
        fprintf(stderr, "\n⚠  No segments produced (model may not match audio language)\n");
    }
    if (r1.processing_time_ms > 0) {
        fprintf(stderr, "\n✅ VAD processing time: %.0f ms\n", r1.processing_time_ms);
    }
    if (br.rtf > 0 && br.rtf < 100) {
        fprintf(stderr, "✅ RTF: %.2f (real-time ratio)\n", br.rtf);
    }

    return pass ? 0 : 1;
}
