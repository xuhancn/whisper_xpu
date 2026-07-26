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

