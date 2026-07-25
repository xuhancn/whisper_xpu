#include "engine.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* audio_path = nullptr;
    int device_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model_path = argv[++i];
        else if (a == "--audio" && i+1 < argc) audio_path = argv[++i];
        else if (a == "--cpu") device_id = -1;
        else if (a == "--help") {
            printf("Usage: bench_vad --model <path> --audio <wav> [--cpu]\n");
            return 0;
        }
    }

    if (!model_path || !audio_path) {
        fprintf(stderr, "Error: --model and --audio are required\n");
        return 1;
    }

    // Load engine
    fprintf(stderr, "Loading engine (device %d)...\n", device_id);
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Audio: %s\n", audio_path);
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
    fprintf(stderr, "\n=== Non-VAD (first 500 chars) ===\n%.*s\n", 500, r0.text.c_str());
    fprintf(stderr, "\n=== VAD (first 500 chars) ===\n%.*s\n", 500, r1.text.c_str());

    return 0;
}
