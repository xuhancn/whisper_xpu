#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// DLL export/import macros for whisper_xpu_core.
//
// The project builds a single core DLL:
//   whisper_xpu_sycl_core.dll (icpx) — engine, device_detect, ggml-sycl, oneDNN
//
// Two export tiers:
//   WHISPER_XPU_SYCL_API — device_detect symbols (WHISPER_XPU_BUILD_SYCL_LIB)
//   WHISPER_XPU_API       — Engine, merge_segments symbols (WHISPER_XPU_BUILD_MAIN_LIB)
//   Both are dllexport when building the DLL, dllimport for consumers.
//
// When WHISPER_XPU_BUILD_SHARED_LIBS is not defined (static build):
//   Both macros are empty — no export/import annotations needed.
// ──────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
  #define WHISPER_XPU_HIDDEN

  // ── Engine / merge_segments API (WHISPER_XPU_BUILD_MAIN_LIB) ──
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_MAIN_LIB
      #define WHISPER_XPU_API __declspec(dllexport)
    #else
      #define WHISPER_XPU_API __declspec(dllimport)
    #endif
  #else
    #define WHISPER_XPU_API
  #endif

  // ── device_detect API (WHISPER_XPU_BUILD_SYCL_LIB) ──
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_SYCL_LIB
      #define WHISPER_XPU_SYCL_API __declspec(dllexport)
    #else
      #define WHISPER_XPU_SYCL_API __declspec(dllimport)
    #endif
  #else
    #define WHISPER_XPU_SYCL_API
  #endif

#else // !_WIN32
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_MAIN_LIB
      #define WHISPER_XPU_API __attribute__((__visibility__("default")))
    #else
      #define WHISPER_XPU_API
    #endif
    #ifdef WHISPER_XPU_BUILD_SYCL_LIB
      #define WHISPER_XPU_SYCL_API __attribute__((__visibility__("default")))
    #else
      #define WHISPER_XPU_SYCL_API
    #endif
    #define WHISPER_XPU_HIDDEN __attribute__((__visibility__("hidden")))
  #else
    #define WHISPER_XPU_API
    #define WHISPER_XPU_SYCL_API
    #define WHISPER_XPU_HIDDEN
  #endif
#endif
