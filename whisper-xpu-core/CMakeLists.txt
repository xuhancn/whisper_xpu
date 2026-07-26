add_library(whisper_xpu_core STATIC
    engine.cpp
    merge_segments.cpp
)

if(HAVE_SYCL)
    set(SYCL_SOURCES sycl_src/device_detect.cpp)
else()
    target_sources(whisper_xpu_core PRIVATE sycl_src/device_detect.cpp)
endif()

target_include_directories(whisper_xpu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src)
target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_BUILD_MAIN_LIB)
target_link_libraries(whisper_xpu_core PUBLIC whisper)

if(HAVE_SYCL)
    # Check whether the project compiler supports -fsycl natively.
    # On Linux with g++ (or other non-Intel compilers) it won't, so we
    # fall back to the Intel SYCL compiler (icpx) found alongside icx.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

    if(SUPPORTS_SYCL OR WIN32)
        # ── Normal path: compiler supports -fsycl ──
        add_library(whisper_xpu_core_sycl STATIC ${SYCL_SOURCES})
        if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
            if(DEFINED ENV{DNNL_INTEL_TOOLSET})
                set_target_properties(whisper_xpu_core_sycl PROPERTIES VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
            else()
                set_target_properties(whisper_xpu_core_sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
            endif()
        endif()
        target_compile_options(whisper_xpu_core_sycl PRIVATE -fsycl)
        target_include_directories(whisper_xpu_core_sycl PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/sycl_src
            ${SYCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include)
        target_compile_definitions(whisper_xpu_core_sycl PRIVATE WHISPER_XPU_HAS_SYCL)
        if(SYCL_LIBRARY_DIR)
            target_link_directories(whisper_xpu_core_sycl PUBLIC ${SYCL_LIBRARY_DIR})
        endif()
        target_link_libraries(whisper_xpu_core_sycl PUBLIC sycl)
    else()
        # ── Fallback: compile SYCL sources with icpx directly ──
        get_filename_component(SYCL_BIN_DIR "${SYCL_COMPILER}" DIRECTORY)
        find_program(SYCL_ICPX icpx PATHS "${SYCL_BIN_DIR}" NO_DEFAULT_PATH)
        if(NOT SYCL_ICPX)
            message(FATAL_ERROR "icpx not found in ${SYCL_BIN_DIR}")
        endif()
        message(STATUS "Using icpx fallback for SYCL compilation: ${SYCL_ICPX}")

        # Gather include flags from the set that the normal path uses
        set(_SYCL_INC_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/sycl_src"
            ${SYCL_INCLUDE_DIR}
            "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include")
        set(_SYCL_INC_FLAGS "")
        foreach(_d IN LISTS _SYCL_INC_DIRS)
            list(APPEND _SYCL_INC_FLAGS "-I${_d}")
        endforeach()

        set(SYCL_OBJS "")
        foreach(_src IN LISTS SYCL_SOURCES)
            get_filename_component(_src_abs "${_src}" ABSOLUTE)
            get_filename_component(_src_name "${_src}" NAME)
            set(_obj "${CMAKE_CURRENT_BINARY_DIR}/icpx_${_src_name}.o")
            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND "${SYCL_ICPX}" -fsycl ${_SYCL_INC_FLAGS}
                    -DWHISPER_XPU_HAS_SYCL -c "${_src_abs}" -o "${_obj}"
                MAIN_DEPENDENCY "${_src_abs}"
                COMMENT "SYCL (icpx): ${_src_name}")
            list(APPEND SYCL_OBJS "${_obj}")
        endforeach()

        add_library(whisper_xpu_core_sycl STATIC ${SYCL_OBJS})
        target_link_libraries(whisper_xpu_core_sycl PUBLIC sycl)
        if(SYCL_LIBRARY_DIR)
            target_link_directories(whisper_xpu_core_sycl PUBLIC ${SYCL_LIBRARY_DIR})
        endif()
    endif()

    target_link_libraries(whisper_xpu_core PRIVATE whisper_xpu_core_sycl)

    # oneDNN static lib is ~420 MB — link directly, don't merge
    if(TARGET DNNL::dnnl AND ONEDNN_STATIC)
        target_link_libraries(whisper_xpu_core PRIVATE DNNL::dnnl)
    endif()
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()

# ── Merge dep .libs (exclude oneDNN — too large) ──
if(WIN32 AND NOT CMAKE_GENERATOR MATCHES "Ninja")
    set(MERGE_DEPS $<TARGET_FILE:whisper> $<TARGET_FILE:ggml> $<TARGET_FILE:ggml-base> $<TARGET_FILE:ggml-cpu>)
    if(TARGET ggml-sycl)
        list(APPEND MERGE_DEPS $<TARGET_FILE:ggml-sycl>)
    endif()
    if(TARGET whisper_xpu_core_sycl)
        list(APPEND MERGE_DEPS $<TARGET_FILE:whisper_xpu_core_sycl>)
    endif()
    add_custom_command(TARGET whisper_xpu_core POST_BUILD
        COMMAND lib.exe /OUT:$<TARGET_FILE:whisper_xpu_core>
            $<TARGET_FILE:whisper_xpu_core>
            ${MERGE_DEPS}
        COMMENT "Merging libraries into whisper_xpu_core.lib")
endif()
