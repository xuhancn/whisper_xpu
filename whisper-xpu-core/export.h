#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// DLL export/import macros for whisper_xpu_core.
//
// When BUILD_SHARED_LIBS is OFF (default, static build):
//   WHISPER_XPU_API is empty — no export/import annotations needed.
//
// When BUILD_SHARED_LIBS is ON:
//   Library build (WHISPER_XPU_BUILD_MAIN_LIB defined):
//     WHISPER_XPU_API → __declspec(dllexport) on Windows
//     WHISPER_XPU_API → __attribute__((visibility("default"))) on Linux/macOS
//   Consumer build:
//     WHISPER_XPU_API → __declspec(dllimport) on Windows
//     WHISPER_XPU_API → (default visibility) on Linux/macOS
// ──────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
  #define WHISPER_XPU_HIDDEN
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_MAIN_LIB
      #define WHISPER_XPU_API __declspec(dllexport)
    #else
      #define WHISPER_XPU_API __declspec(dllimport)
    #endif
  #else
    #define WHISPER_XPU_API
  #endif
#else // !_WIN32
  #ifdef WHISPER_XPU_BUILD_SHARED_LIBS
    #ifdef WHISPER_XPU_BUILD_MAIN_LIB
      #define WHISPER_XPU_API __attribute__((__visibility__("default")))
    #else
      #define WHISPER_XPU_API
    #endif
    #define WHISPER_XPU_HIDDEN __attribute__((__visibility__("hidden")))
  #else
    #define WHISPER_XPU_API
    #define WHISPER_XPU_HIDDEN
  #endif
#endif
