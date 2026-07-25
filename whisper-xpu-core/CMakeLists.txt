# ── Main core library ──
add_library(whisper_xpu_core STATIC
    engine.cpp
    merge_segments.cpp
)

# device_detect.cpp goes into whisper_xpu_core when SYCL is OFF,
# or into whisper_xpu_core_sycl when SYCL is ON (compiled with Intel compiler + -fsycl)
if(HAVE_SYCL)
    set(SYCL_SOURCES sycl_src/device_detect.cpp)
else()
    target_sources(whisper_xpu_core PRIVATE sycl_src/device_detect.cpp)
endif()

target_include_directories(whisper_xpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
)

target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_BUILD_MAIN_LIB)
target_link_libraries(whisper_xpu_core PUBLIC whisper)

# ── SYCL sub-library ──
if(HAVE_SYCL)
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
        ${SYCL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include)
    target_compile_definitions(whisper_xpu_core_sycl PRIVATE WHISPER_XPU_HAS_SYCL)

    if(SYCL_LIBRARY_DIR)
        target_link_directories(whisper_xpu_core_sycl PUBLIC ${SYCL_LIBRARY_DIR})
    endif()
    target_link_libraries(whisper_xpu_core_sycl PUBLIC sycl)

    target_link_libraries(whisper_xpu_core PRIVATE whisper_xpu_core_sycl)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()

# ── Merge all dep .libs into one whisper_xpu_core.lib ──
if(WIN32 AND NOT CMAKE_GENERATOR MATCHES "Ninja")
    set(MERGE_DEPS
        $<TARGET_FILE:whisper>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE:ggml-base>
        $<TARGET_FILE:ggml-cpu>
    )
    if(TARGET ggml-sycl)
        list(APPEND MERGE_DEPS $<TARGET_FILE:ggml-sycl>)
    endif()
    if(TARGET whisper_xpu_core_sycl)
        list(APPEND MERGE_DEPS $<TARGET_FILE:whisper_xpu_core_sycl>)
    endif()
    if(TARGET DNNL::dnnl AND ONEDNN_STATIC)
        list(APPEND MERGE_DEPS $<TARGET_FILE:DNNL::dnnl>)
    endif()

    add_custom_command(TARGET whisper_xpu_core POST_BUILD
        COMMAND lib.exe /OUT:$<TARGET_FILE:whisper_xpu_core>
            $<TARGET_FILE:whisper_xpu_core>
            ${MERGE_DEPS}
        COMMENT "Merging libraries into whisper_xpu_core.lib"
    )
endif()
