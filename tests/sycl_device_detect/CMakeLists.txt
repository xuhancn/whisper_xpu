add_executable(test_sycl_device_detect
    test_sycl_device_detect.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
)
target_include_directories(test_sycl_device_detect PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
target_link_libraries(test_sycl_device_detect PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME sycl_device_detect/basics COMMAND test_sycl_device_detect)

if(HAVE_SYCL)
    set(TEST_SYCL_SOURCES
        test_sycl_device_detect.cpp
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
    )
    set(TEST_SYCL_INC_DIRS
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
        ${SYCL_INCLUDE_DIR}
    )

    # Check whether the project compiler supports -fsycl (icpx) natively.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

    if(SUPPORTS_SYCL OR WIN32)
        # ── Normal path: compiler supports -fsycl ──
        add_executable(test_sycl_device_detect_sycl ${TEST_SYCL_SOURCES})
        target_include_directories(test_sycl_device_detect_sycl PRIVATE ${TEST_SYCL_INC_DIRS})
        target_compile_definitions(test_sycl_device_detect_sycl PRIVATE WHISPER_XPU_HAS_SYCL)
        target_compile_options(test_sycl_device_detect_sycl PRIVATE -fsycl)
        if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
            if(DEFINED ENV{DNNL_INTEL_TOOLSET})
                set_target_properties(test_sycl_device_detect_sycl PROPERTIES
                    VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
            else()
                set_target_properties(test_sycl_device_detect_sycl PROPERTIES
                    VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
            endif()
        endif()
    else()
        # ── Fallback: compile SYCL sources with icpx directly ──
        get_filename_component(SYCL_BIN_DIR "${SYCL_COMPILER}" DIRECTORY)
        find_program(SYCL_ICPX icpx PATHS "${SYCL_BIN_DIR}" NO_DEFAULT_PATH)
        if(NOT SYCL_ICPX)
            message(FATAL_ERROR "icpx not found in ${SYCL_BIN_DIR}")
        endif()
        message(STATUS "Using icpx fallback for SYCL compilation: ${SYCL_ICPX}")

        set(_SYCL_INC_FLAGS "")
        foreach(_d IN LISTS TEST_SYCL_INC_DIRS)
            list(APPEND _SYCL_INC_FLAGS "-I${_d}")
        endforeach()

        set(SYCL_OBJS "")
        foreach(_src IN LISTS TEST_SYCL_SOURCES)
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

        add_executable(test_sycl_device_detect_sycl ${SYCL_OBJS})
    endif()

    target_link_libraries(test_sycl_device_detect_sycl PRIVATE GTest::gtest GTest::gtest_main)
    if(SYCL_LIBRARY_DIR)
        target_link_directories(test_sycl_device_detect_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    target_link_libraries(test_sycl_device_detect_sycl PRIVATE sycl)
    add_test(NAME sycl_device_detect/sycl COMMAND test_sycl_device_detect_sycl)
endif()
