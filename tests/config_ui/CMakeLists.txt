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

# ── SYCL-dependent test — linked against whisper_xpu_sycl_core import lib ──
if(HAVE_SYCL)
    add_executable(test_config_sycl
        test_config_sycl.cpp
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
    target_link_libraries(test_config_sycl PRIVATE
        GTest::gtest GTest::gtest_main
        whisper_xpu_sycl_core
        sycl
    )
    if(SYCL_LIBRARY_DIR)
        target_link_directories(test_config_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    add_test(NAME config_ui/sycl COMMAND test_config_sycl)
endif()
