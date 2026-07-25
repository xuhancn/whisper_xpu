# ── Non-SYCL device_detect test (compiles with MSVC) ──
add_executable(test_device_detect
    test_device_detect.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
)
target_include_directories(test_device_detect PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
target_compile_definitions(test_device_detect PRIVATE
    # Intentionally NOT defining WHISPER_XPU_HAS_SYCL so the SYCL paths
    # are excluded and the test compiles with MSVC alone.
)
add_test(NAME test_device_detect COMMAND test_device_detect)

# ── SYCL device_detect test (needs Intel compiler, only when HAVE_SYCL) ──
if(HAVE_SYCL)
    add_executable(test_device_detect_sycl
        test_device_detect.cpp
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
    )
    target_include_directories(test_device_detect_sycl PRIVATE
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_definitions(test_device_detect_sycl PRIVATE WHISPER_XPU_HAS_SYCL)
    target_compile_options(test_device_detect_sycl PRIVATE -fsycl)

    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set_target_properties(test_device_detect_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set_target_properties(test_device_detect_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        endif()
    endif()

    if(SYCL_LIBRARY_DIR)
        target_link_directories(test_device_detect_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    target_link_libraries(test_device_detect_sycl PRIVATE sycl)

    add_test(NAME test_device_detect_sycl COMMAND test_device_detect_sycl)
endif()

# Enable CTest
enable_testing()
