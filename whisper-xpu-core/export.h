#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// DLL export/import macros for whisper_xpu_core.
//
// The project builds two DLLs:
//   1. whisper_xpu_cpu_core.dll  (MSVC) — Engine, merge, no SYCL
//   2. whisper_xpu_sycl_core.dll (icpx) — device_detect, ggml-sycl, oneDNN
//
// For the CPU DLL:
//   Library build  (WHISPER_XPU_BUILD_MAIN_LIB):  WHISPER_XPU_API → dllexport
//   Consumer build:                               WHISPER_XPU_API → dllimport
//
// For the SYCL DLL:
//   Library build  (WHISPER_XPU_BUILD_SYCL_LIB):  WHISPER_XPU_SYCL_API → dllexport
//   Consumer build:                               WHISPER_XPU_SYCL_API → dllimport
//
// When WHISPER_XPU_BUILD_SHARED_LIBS is not defined (static build):
//   Both macros are empty — no export/import annotations needed.
// ──────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
  #define WHISPER_XPU_HIDDEN

  // ── CPU DLL (WHISPER_XPU_BUILD_MAIN_LIB) ──
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_MAIN_LIB
      #define WHISPER_XPU_API __declspec(dllexport)
    #else
      #define WHISPER_XPU_API __declspec(dllimport)
    #endif
  #else
    #define WHISPER_XPU_API
  #endif

  // ── SYCL DLL (WHISPER_XPU_BUILD_SYCL_LIB) ──
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
