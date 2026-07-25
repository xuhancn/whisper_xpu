# ── Main core library (MSVC) ──
add_library(whisper_xpu_core STATIC engine.cpp)

target_include_directories(whisper_xpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# whisper.cpp headers
target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_link_libraries(whisper_xpu_core PUBLIC
    whisper
)

# ── SYCL sub-library (Intel C++ Compiler + -fsycl) ──
if(HAVE_SYCL)
    file(GLOB_RECURSE SYCL_SOURCES sycl_src/*.cpp)
    add_library(whisper_xpu_core_sycl STATIC ${SYCL_SOURCES})

    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set_target_properties(whisper_xpu_core_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set_target_properties(whisper_xpu_core_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        endif()
    endif()

    target_compile_options(whisper_xpu_core_sycl PRIVATE -fsycl)
    target_include_directories(whisper_xpu_core_sycl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
        ${SYCL_INCLUDE_DIR})
    target_compile_definitions(whisper_xpu_core_sycl PRIVATE WHISPER_XPU_HAS_SYCL)

    if(SYCL_LIBRARY_DIR)
        target_link_directories(whisper_xpu_core_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    target_link_libraries(whisper_xpu_core_sycl PUBLIC sycl)

    target_link_libraries(whisper_xpu_core PRIVATE whisper_xpu_core_sycl)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()
