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
target_link_libraries(whisper_xpu_cpu_core PRIVATE whisper_xpu_sycl_core)

if(WIN32)
    target_link_libraries(whisper_xpu_cpu_core PRIVATE winmm)
endif()

# ═════════════════════════════════════════════════════════════════════
# xpu_device_prop.dll — reference probe (icpx, SHARED)
#   Exact match of xpu_kernel_optimizer/projection/scripts/CMakeLists.txt
#   Source: xpuDeviceProp.cpp (zero modifications)
# ═════════════════════════════════════════════════════════════════════

add_library(xpu_device_prop SHARED
    sycl_src/xpuDeviceProp.cpp       # unmodified reference code
    sycl_src/xpu_probe_seh.cpp       # SEH wrapper — calls the above safely
)
target_compile_options(xpu_device_prop PRIVATE -fsycl)
target_link_options(xpu_device_prop PRIVATE -fsycl)
set_target_properties(xpu_device_prop PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX ""
    # Force icpx (clang-based SYCL compiler) — matches original project
    CXX_COMPILER "icpx"
)

target_include_directories(xpu_device_prop PRIVATE
    ${SYCL_INCLUDE_DIR}
)

if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
    if(DEFINED ENV{DNNL_INTEL_TOOLSET})
        set_target_properties(xpu_device_prop PROPERTIES
            VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
    else()
        set_target_properties(xpu_device_prop PROPERTIES
            VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
    endif()
endif()

if(SYCL_LIBRARY_DIR)
    target_link_directories(xpu_device_prop PRIVATE ${SYCL_LIBRARY_DIR})
endif()
target_link_libraries(xpu_device_prop PRIVATE sycl)

# Delay-load sycl8.dll so the probe DLL can be loaded before
# ZES_ENABLE_SYSMAN=1 — only triggers sycl8.dll on first SYCL call.
if(WIN32)
    target_link_options(xpu_device_prop PRIVATE /DELAYLOAD:sycl8.dll)
    target_link_libraries(xpu_device_prop PRIVATE delayimp)
endif()

# Copy to app output dir
add_custom_command(TARGET xpu_device_prop POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:xpu_device_prop>
        $<TARGET_FILE_DIR:whisper_xpu_app>
    COMMENT "Copying xpu_device_prop.dll to app output directory"
)

# Loader EXE (MSVC, no SYCL) — sets ZES_ENABLE_SYSMAN then LoadLibrary
add_executable(xpu_probe_loader sycl_src/probe_loader.cpp)
target_compile_options(xpu_probe_loader PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_custom_command(TARGET xpu_probe_loader POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:xpu_probe_loader>
        $<TARGET_FILE_DIR:whisper_xpu_app>
    COMMENT "Copying xpu_probe_loader.exe to app output directory"
)
