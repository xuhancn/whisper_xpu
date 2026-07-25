add_library(whisper_xpu_core STATIC
    engine.cpp
)

if(HAVE_SYCL)
    # device_detect.cpp uses SYCL APIs directly and needs the Intel
    # C++ Compiler.  VS_TOOLSET_OVERRIDE (CMake 3.22+) lets us use
    # a different toolset for a single source file within an MSVC target.
    target_sources(whisper_xpu_core PRIVATE device_detect.cpp)
    target_include_directories(whisper_xpu_core PRIVATE ${SYCL_INCLUDE_DIR})
    target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_HAS_SYCL)
    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set_source_files_properties(device_detect.cpp PROPERTIES
                VS_TOOLSET_OVERRIDE "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set_source_files_properties(device_detect.cpp PROPERTIES
                VS_TOOLSET_OVERRIDE "Intel C++ Compiler 2025")
        endif()
    endif()
endif()

target_include_directories(whisper_xpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# whisper.cpp headers
target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_link_libraries(whisper_xpu_core PUBLIC
    whisper    # from whisper.cpp's CMake
)

if(HAVE_SYCL)
    target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_HAS_SYCL)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()
