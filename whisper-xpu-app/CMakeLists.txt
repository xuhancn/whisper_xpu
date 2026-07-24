# ── SYCL device detection library ──
# Compiled with the Intel C++ Compiler on Windows (needed for SYCL headers).
# Other source files compile with the default MSVC compiler.
if(HAVE_SYCL)
    add_library(whisper_xpu_sycl_detect STATIC
        device_detect.cpp
    )
    target_include_directories(whisper_xpu_sycl_detect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_definitions(whisper_xpu_sycl_detect PRIVATE WHISPER_XPU_HAS_SYCL)
    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        set_target_properties(whisper_xpu_sycl_detect PROPERTIES
            VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025"
        )
    endif()
else()
    add_library(whisper_xpu_sycl_detect STATIC
        device_detect.cpp
    )
    target_include_directories(whisper_xpu_sycl_detect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)

target_link_libraries(whisper_xpu_app PRIVATE
    whisper_xpu_core
    whisper_xpu_sycl_detect
    wx::core          # wxWidgets GUI library (depends on wx::base)
    portaudio
)

if(WIN32)
    target_link_libraries(whisper_xpu_app PRIVATE
        winmm
        ole32
        uuid
        comctl32
        rpcrt4
    )
    # whisper_xpu_sycl_detect is compiled with Intel C++ Compiler on Windows
    # and uses SYCL runtime APIs directly (sycl::platform::get_platforms, etc.).
    # When linked into an MSVC-compiled target, the MSVC linker needs to find:
    #   1. sycl.lib      - SYCL runtime library
    #   2. libircmt.lib  - Intel OpenMP runtime (brought in by Intel C++ compiler)
    # Add the Intel compiler library directory to the search path
    # and link against sycl.lib explicitly.
    if(HAVE_SYCL AND SYCL_LIBRARY_DIR)
        target_link_directories(whisper_xpu_app PRIVATE ${SYCL_LIBRARY_DIR})
        target_link_libraries(whisper_xpu_app PRIVATE sycl)
    endif()
endif()
