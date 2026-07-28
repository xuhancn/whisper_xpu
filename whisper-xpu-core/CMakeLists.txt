# ═════════════════════════════════════════════════════════════════════
# whisper_xpu_sycl_core.dll  (icpx-compiled SHARED)
#   Single core DLL: engine, merge_segments, device enumeration,
#   ggml-sycl GPU operators, oneDNN — all in one place.
# ═════════════════════════════════════════════════════════════════════

add_library(whisper_xpu_sycl_core SHARED
    sycl_src/device_detect.cpp
    engine.cpp
    merge_segments.cpp
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

# -fsycl is added to BOTH compile and link (see link block below).  icpx honors
# it at compile (emits SPIR-V alongside host code).  But at the LINK step, under
# the VS generator + Intel MSBuild toolset, the linker is lld-link, which IGNORES
# -fsycl as an unknown CLI flag (you see "ignoring unknown argument '-fsycl'" in
# the link log).  Without the device-code post-link (sycl-post-link +
# clang-offload-bundler) the /WHOLEARCHIVE'd ggml-sycl compute kernels never get
# bundled into a device image inside this DLL, and the first GPU op dies at
# runtime: "No kernel named im2col_sycl<half> was found".
#
# The Windows mechanism that triggers sycl-post-link is the EnableSyclOffload
# MSBuild metadata: the Intel toolset's link task reads it and runs the post-link.
# This is NOT a PlatformToolset or a hardcoded version — it is a per-source
# compile metadata flag, the same role -fsycl-as-link-driver plays on Linux/Ninja.
# PyTorch's FindSYCLToolkit (and our cmake/FindSYCLToolkit.cmake) avoids it only
# because it targets icx-as-linker; on Windows under the VS generator there is no
# CLI equivalent lld-link honors, so the metadata is mandatory here.
set_source_files_properties(
    sycl_src/device_detect.cpp engine.cpp merge_segments.cpp
    PROPERTIES VS_SETTINGS "EnableSyclOffload=true"
)

target_compile_options(whisper_xpu_sycl_core PRIVATE -fsycl -fno-delayed-template-parsing)

# Under the VS generator the SYCL core MUST compile with the Intel oneAPI
# compiler: MSVC's cl.exe does not understand -fsycl.  We do NOT hardcode a
# toolset version here — the name comes from the DNNL_INTEL_TOOLSET env var
# (set by the build .bat after setvars.bat), the same source ggml-sycl and
# oneDNN consult.  If it is unset, fail loudly rather than guess a year.
if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
    if(DEFINED ENV{DNNL_INTEL_TOOLSET})
        set_target_properties(whisper_xpu_sycl_core PROPERTIES
            VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
    else()
        message(FATAL_ERROR
            "DNNL_INTEL_TOOLSET not set.  Under the VS generator the SYCL core "
            "needs the Intel oneAPI toolset (cl.exe cannot compile -fsycl).  "
            "Run setvars.bat, then:  set DNNL_INTEL_TOOLSET=Intel C++ Compiler 2026")
    endif()
endif()

# SYCL runtime
if(SYCL_LIBRARY_DIR)
    target_link_directories(whisper_xpu_sycl_core PUBLIC ${SYCL_LIBRARY_DIR})
endif()
target_link_libraries(whisper_xpu_sycl_core PUBLIC sycl whisper)

if(WIN32)
    target_link_libraries(whisper_xpu_sycl_core PRIVATE winmm)
endif()

# ── ggml-sycl: link statically, export all its symbols ──
# ggml-sycl contains all GPU kernel implementations.  We embed it
# into the SYCL DLL via /WHOLEARCHIVE so that ggml_backend_sycl_*
# symbols are available to engine.cpp within the same DLL.
target_link_libraries(whisper_xpu_sycl_core PRIVATE ggml-sycl)

if(WIN32)
    # /WHOLEARCHIVE embeds all ggml-sycl objects into the DLL.  The
    # WINDOWS_EXPORT_ALL_SYMBOLS property only covers the DLL's own
    # objects, so we export the ggml_backend_sycl_* API explicitly.
    target_link_options(whisper_xpu_sycl_core PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:ggml-sycl>"
        /EXPORT:ggml_backend_sycl_init
        /EXPORT:ggml_backend_is_sycl
        /EXPORT:ggml_backend_sycl_buffer_type
        /EXPORT:ggml_backend_sycl_split_buffer_type
        /EXPORT:ggml_backend_sycl_comm_init
        /EXPORT:ggml_backend_sycl_comm_free
        /EXPORT:ggml_backend_sycl_comm_allreduce_tensor
        /EXPORT:ggml_backend_sycl_host_buffer_type
        /EXPORT:ggml_backend_sycl_print_sycl_devices
        /EXPORT:ggml_backend_sycl_get_gpu_list
        /EXPORT:ggml_backend_sycl_get_device_description
        /EXPORT:ggml_backend_sycl_get_device_count
        /EXPORT:ggml_backend_sycl_get_device_memory
        /EXPORT:ggml_backend_sycl_reg
    )
    # WINDOWS_EXPORT_ALL_SYMBOLS still needed for device_detect symbols
    # and any internal symbols from the DLL's own objects.
    set_target_properties(whisper_xpu_sycl_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# -fsycl on the LINK (not just compile) is what makes xilink run the SYCL
# device-code post-link over the /WHOLEARCHIVE'd ggml-sycl objects, bundling
# their compute kernels (im2col, matmul, …) into a device image inside this
# DLL.  Without -fsycl on link the link is a plain host link and every compute
# kernel is missing at runtime ("No kernel named im2col_sycl<half> was found").
#
# Delay-load sycl9.dll (oneAPI 2026.1+; was sycl8.dll through 2025.x) so the
# DLL can load even when the SYCL runtime is absent — functions return 0 /
# empty gracefully.
if(WIN32)
    target_link_options(whisper_xpu_sycl_core PRIVATE
        -fsycl
        /DELAYLOAD:sycl9.dll
    )
    target_link_libraries(whisper_xpu_sycl_core PRIVATE delayimp)
endif()

# oneDNN (static, not merged — too large at ~420 MB)
if(TARGET DNNL::dnnl AND ONEDNN_STATIC)
    target_link_libraries(whisper_xpu_sycl_core PRIVATE DNNL::dnnl)
endif()
