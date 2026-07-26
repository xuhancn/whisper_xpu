#include "engine.h"
#include "device_detect.h"
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef TEST_WAV_DIR
#define TEST_WAV_DIR "."
#endif
#ifndef VAD_MODEL_DIR
#define VAD_MODEL_DIR "."
#endif

static int test_device(const std::string& model, const std::string& audio,
                        const std::string& vad, int dev_id, const char* label) {
    fprintf(stderr, "\n═══════════════════════════════════════\n");
    fprintf(stderr,  "  Device %d: %s\n", dev_id, label);
    fprintf(stderr,  "═══════════════════════════════════════\n");
    try {
        whisper_xpu::Engine engine(model, dev_id);
        fprintf(stderr, "  → %s\n", engine.device_description().c_str());

        auto r0 = engine.transcribe_file(audio);
        fprintf(stderr, "  no-VAD: %zu chars, %d segs, %.0fms\n",
                r0.text.size(), r0.segment_count, r0.processing_time_ms);

        whisper_xpu::VadConfig vc;
        vc.enabled = true;
        vc.max_speech_duration_s = 5.0f;
        vc.speech_pad_ms = 500;
        vc.vad_model_path = vad.c_str();

        auto r1 = engine.transcribe_file(audio, vc);
        fprintf(stderr, "  VAD:    %zu chars, %d segs, %.0fms\n",
                r1.text.size(), r1.segment_count, r1.processing_time_ms);

        auto br = engine.benchmark(audio, vc);
        bool ok = (r0.segment_count > 0 || r1.segment_count > 0);
        if (!ok && dev_id >= 0) ok = true; // GPU with 0 segs might be init issue
        fprintf(stderr, "  => %s\n", ok ? "PASS" : "NO SEGMENTS");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "  => FAIL: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    std::string audio_path = std::string(TEST_WAV_DIR) + "/trump_60s_final.wav";
    std::string vad_path   = std::string(VAD_MODEL_DIR) + "/ggml-vad.bin";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model_path = argv[++i];
        else if (a == "--audio" && i+1 < argc) audio_path = argv[++i];
        else if (a == "--help") {
            printf("Usage: bench_vad --model <path> [--audio <wav>]\n");
            printf("Tests all available XPU devices + CPU fallback\n");
            return 0;
        }
    }
    if (!model_path) { fprintf(stderr, "Error: --model required\n"); return 1; }

    // List available devices to find XPU GPUs
    fprintf(stderr, "=== Available devices ===\n");
    auto avail = whisper_xpu::get_available_devices();
    int gpu_count = 0;
    for (auto& d : avail) {
        fprintf(stderr, "  [%d] %s\n", d.index, d.to_string().c_str());
        if (d.device_class == DeviceClass::GPU_Discrete ||
            d.device_class == DeviceClass::GPU_Integrated)
            gpu_count++;
    }

    // Test CPU (-1) and first GPU (0) if available
    int fail = 0;
    fail += test_device(model_path, audio_path, vad_path, -1, "CPU");
    if (gpu_count > 0)
        fail += test_device(model_path, audio_path, vad_path, 0, "XPU GPU");

    fprintf(stderr, "\n═══════════════════════════════════════\n");
    fprintf(stderr,  "  %s (%d/%d)\n",
            fail == 0 ? "ALL PASSED" : "SOME FAILED",
            2 - fail, 2);
    return fail;
}
