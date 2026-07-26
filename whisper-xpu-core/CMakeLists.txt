# ═════════════════════════════════════════════════════════════════════
# whisper_xpu_sycl_core.dll  (icpx-compiled SHARED)
#   All SYCL code: device enumeration + ggml-sycl GPU operators + oneDNN
# ═════════════════════════════════════════════════════════════════════

add_library(whisper_xpu_sycl_core SHARED
    sycl_src/device_detect.cpp
)

target_compile_definitions(whisper_xpu_sycl_core PRIVATE
    WHISPER_XPU_BUILD_SYCL_LIB
    WHISPER_XPU_BUILD_SHARED_LIBS
    WHISPER_XPU_HAS_SYCL
)

target_include_directories(whisper_xpu_sycl_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
    ${SYCL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_compile_options(whisper_xpu_sycl_core PRIVATE -fsycl)

# Intel C++ Compiler toolset on Windows
if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
    if(DEFINED ENV{DNNL_INTEL_TOOLSET})
        set_target_properties(whisper_xpu_sycl_core PROPERTIES
            VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
    else()
        set_target_properties(whisper_xpu_sycl_core PROPERTIES
            VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
    endif()
endif()

# SYCL runtime
if(SYCL_LIBRARY_DIR)
    target_link_directories(whisper_xpu_sycl_core PUBLIC ${SYCL_LIBRARY_DIR})
endif()
target_link_libraries(whisper_xpu_sycl_core PUBLIC sycl)

# ── ggml-sycl: link statically, export all its symbols ──
# ggml-sycl contains all GPU kernel implementations.  We embed it
# into the SYCL DLL so that the CPU DLL imports ggml_backend_sycl_*
# from here rather than linking ggml-sycl directly (which would pull
# SYCL runtime dependencies into the MSVC binary).
target_link_libraries(whisper_xpu_sycl_core PRIVATE ggml-sycl)

if(WIN32)
    # /WHOLEARCHIVE forces every object in ggml-sycl.lib into the DLL.
    # Without it the linker drops unreferenced objects and the GPU
    # operators wouldn't be available at runtime.
    target_link_options(whisper_xpu_sycl_core PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:ggml-sycl>"
    )
    # Auto-generate a .def file so that all symbols (including
    # ggml_backend_sycl_*) are exported from the DLL.
    set_target_properties(whisper_xpu_sycl_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Delay-load sycl8.dll so the DLL can load even when the SYCL
# runtime is absent — functions return 0 / empty gracefully.
if(WIN32)
    target_link_options(whisper_xpu_sycl_core PRIVATE
        /DELAYLOAD:sycl8.dll
    )
    target_link_libraries(whisper_xpu_sycl_core PRIVATE delayimp)
endif()

# oneDNN (static, not merged — too large at ~420 MB)
if(TARGET DNNL::dnnl AND ONEDNN_STATIC)
    target_link_libraries(whisper_xpu_sycl_core PRIVATE DNNL::dnnl)
endif()

# ═════════════════════════════════════════════════════════════════════
# whisper_xpu_cpu_core.dll  (MSVC-compiled SHARED)
#   Engine, merge_segments — no SYCL compilation, no icpx.
#   Imports ggml_backend_sycl_* from whisper_xpu_sycl_core.dll.
# ═════════════════════════════════════════════════════════════════════

add_library(whisper_xpu_cpu_core SHARED
    engine.cpp
    merge_segments.cpp
)

target_compile_definitions(whisper_xpu_cpu_core PRIVATE
    WHISPER_XPU_BUILD_MAIN_LIB
    WHISPER_XPU_BUILD_SHARED_LIBS
    WHISPER_XPU_HAS_SYCL
)

target_include_directories(whisper_xpu_cpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(whisper_xpu_cpu_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

# MSVC compile options
target_compile_options(whisper_xpu_cpu_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

# ── Dependencies ──
# whisper + ggml are linked statically (symbols resolved at link time,
# not re-exported from this DLL).
target_link_libraries(whisper_xpu_cpu_core PUBLIC whisper)

# Import ggml_backend_sycl_* and device_detect symbols from the SYCL DLL.
# /DELAYLOAD so the SYCL DLL doesn't load at process startup — ZES_ENABLE_SYSMAN
# must be set first (done in main.cpp before any SYCL call).
target_link_libraries(whisper_xpu_cpu_core PRIVATE whisper_xpu_sycl_core)
if(WIN32)
    target_link_options(whisper_xpu_cpu_core PRIVATE
        /DELAYLOAD:whisper_xpu_sycl_core.dll
    )
    target_link_libraries(whisper_xpu_cpu_core PRIVATE delayimp)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_cpu_core PRIVATE winmm)
endif()
