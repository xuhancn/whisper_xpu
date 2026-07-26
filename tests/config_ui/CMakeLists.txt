add_executable(test_config_ui
    test_config.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/audio_capture.cpp
)
target_include_directories(test_config_ui PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)
target_link_libraries(test_config_ui PRIVATE
    GTest::gtest GTest::gtest_main
    portaudio
)
add_test(NAME config_ui/audio COMMAND test_config_ui)

# ── SYCL-dependent test — compiled with Intel toolset ──
# Follows the same pattern as test_sycl_device_detect_sycl
if(HAVE_SYCL)
    add_executable(test_config_sycl
        test_config_sycl.cpp
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
    )
    target_include_directories(test_config_sycl PRIVATE
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/whisper-xpu-app
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_definitions(test_config_sycl PRIVATE WHISPER_XPU_HAS_SYCL)
    target_compile_options(test_config_sycl PRIVATE -fsycl -DWIN32)
    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set_target_properties(test_config_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set_target_properties(test_config_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        endif()
    endif()
    target_link_libraries(test_config_sycl PRIVATE
        GTest::gtest GTest::gtest_main
        sycl
    )
    if(SYCL_LIBRARY_DIR)
        target_link_directories(test_config_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    add_test(NAME config_ui/sycl COMMAND test_config_sycl)
endif()
