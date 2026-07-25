#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// whisper_xpu_core — single public header for the whisper-xpu inference library
//
// Drop the release/ folder into your project and:
//   #include "whisper_xpu_core.h"
//   link: whisper_xpu_core.lib + sycl.dll + dnnl.dll + OpenCL.lib
// ──────────────────────────────────────────────────────────────────────────────

#include "engine.h"
#include "device_detect.h"
#include "merge_segments.h"

// Re-export whisper.h for advanced usage (model params, sampling, etc.)
#include "whisper.h"

// Re-export ggml-sycl.h for SYCL backend queries
#include "ggml-sycl.h"
