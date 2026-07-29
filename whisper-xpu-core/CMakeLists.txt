# ═════════════════════════════════════════════════════════════════════
# whisper_xpu_sycl_core.dll
#   Core DLL: engine, merge_segments, device enumeration.
#
#   GPU kernels live ENTIRELY in ggml-sycl, built by whisper.cpp's own
#   CMake.  Under the Visual Studio generator whisper.cpp sets
#   VS_PLATFORM_TOOLSET = "Intel C++ Compiler 2025" on the ggml-sycl
#   target, so xilink (not lld-link) drives its link and runs
#   sycl-post-link + clang-offload-bundler → the SPIR-V device image is
#   bundled automatically.  This DLL just links ggml-sycl; no manual
#   -fsycl / EnableSyclOffload / /WHOLEARCHIVE / /DELAYLOAD hacks.
#
#   The DLL's own sources (device_detect.cpp, engine.cpp,
#   merge_segments.cpp) are plain C++ over ggml-sycl's C API
#   (ggml_backend_sycl_*); they deliberately do NOT include
#   <sycl/sycl.hpp>, so they compile cleanly with MSVC — no -fsycl, no
#   Intel toolset needed for this target.
# ═════════════════════════════════════════════════════════════════════

add_library(whisper_xpu_sycl_core SHARED
    sycl_src/device_detect.cpp
    engine.cpp
    merge_segments.cpp
    zh_t2s.cpp
)

target_compile_definitions(whisper_xpu_sycl_core PRIVATE
    WHISPER_XPU_BUILD_MAIN_LIB      # Engine, merge_segments exports
    WHISPER_XPU_BUILD_SYCL_LIB      # device_detect exports
    WHISPER_XPU_BUILD_SHARED_LIBS
    WHISPER_XPU_HAS_SYCL
)

target_include_directories(whisper_xpu_sycl_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(whisper_xpu_sycl_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
    ${SYCL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

# whisper.cpp's libraries are SHARED DLLs produced by the Ninja+icx
# ExternalProject (see top-level CMakeLists.txt).  Linking the imported
# `whisper` target transitively pulls ggml → ggml-base/ggml-cpu/ggml-sycl.
# The SYCL kernel device image AND oneDNN both live inside ggml-sycl.dll, so
# this MSVC-built core needs no -fsycl, no sycl.lib, no DNNL link — it is a
# thin host wrapper over ggml-sycl's C API.
target_link_libraries(whisper_xpu_sycl_core PUBLIC whisper)

if(WIN32)
    target_link_libraries(whisper_xpu_sycl_core PRIVATE winmm)
    # Export the DLL's own symbols (device_detect etc.) and ggml-sycl's
    # backend API that the app reaches through this DLL.
    set_target_properties(whisper_xpu_sycl_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()
