add_library(whisper_xpu_core STATIC
    engine.cpp
)

if(HAVE_SYCL)
    # ── SYCL source files ──
    # All files in sycl_src/ are compiled with Intel C++ Compiler and -fsycl.
    file(GLOB SYCL_SOURCES sycl_src/*.cpp)
    target_sources(whisper_xpu_core PRIVATE ${SYCL_SOURCES})
    target_include_directories(whisper_xpu_core PRIVATE ${SYCL_INCLUDE_DIR})
    target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_HAS_SYCL)

    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set(_sycl_toolset "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set(_sycl_toolset "Intel C++ Compiler 2025")
        endif()
        foreach(_src ${SYCL_SOURCES})
            set_source_files_properties(${_src} PROPERTIES
                VS_TOOLSET_OVERRIDE "${_sycl_toolset}"
                COMPILE_FLAGS "-fsycl")
        endforeach()
    else()
        foreach(_src ${SYCL_SOURCES})
            set_source_files_properties(${_src} PROPERTIES
                COMPILE_FLAGS "-fsycl")
        endforeach()
    endif()

    # sycl.lib for SYCL runtime symbols
    if(SYCL_LIBRARY_DIR)
        target_link_directories(whisper_xpu_core PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    target_link_libraries(whisper_xpu_core PRIVATE sycl)
endif()

target_include_directories(whisper_xpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_link_libraries(whisper_xpu_core PUBLIC
    whisper
)

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()
