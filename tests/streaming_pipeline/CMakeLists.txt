# Headless streaming-pipeline unit test.
#
# Compiles the SAME scheduler sources the app uses (transcription_scheduler,
# sched_log, audio_capture) directly into a console exe — no wx, no GUI — and
# links the icpx-built whisper_xpu_sycl_core.dll for Engine.  The test feeds a
# WAV via start_no_capture()+feed_audio() instead of a microphone.

add_executable(test_pipeline
    test_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/transcription_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/sched_log.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/audio_capture.cpp
)

target_include_directories(test_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app        # "src/transcription_scheduler.h", "../audio_capture.h"
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)

target_compile_options(test_pipeline PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

target_compile_definitions(test_pipeline PRIVATE
    TEST_WAV_DIR="${CMAKE_SOURCE_DIR}/tests/bench_vad"
)

target_link_libraries(test_pipeline PRIVATE
    whisper_xpu_sycl_core
    portaudio
)

if(WIN32)
    target_link_libraries(test_pipeline PRIVATE winmm ole32 uuid)
endif()

# ── Copy the SYCL core DLL next to the test exe ──
add_custom_command(TARGET test_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:test_pipeline>
    COMMENT "Copying whisper_xpu_sycl_core.dll to test output directory"
)

# ── Copy Intel oneAPI runtime DLLs next to the test exe ──
# oneAPI 2026.1: sycl8.dll → sycl9.dll; tcm.dll moved to tcm/latest/bin.
# Every copy is EXISTS-guarded so a relocated DLL never fails the build.
# run_gpu_test.bat sets setvars, so PATH covers any DLL not copied here.
if(WIN32 AND SYCL_LIBRARY_DIR)
    get_filename_component(_sycl_bin "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll
        ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll)
    foreach(_dll ${_rt_dlls})
        if(EXISTS "${_sycl_bin}/${_dll}")
            add_custom_command(TARGET test_pipeline POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_sycl_bin}/${_dll}"
                    $<TARGET_FILE_DIR:test_pipeline>
                COMMENT "Copying ${_dll} to test output directory")
        else()
            message(STATUS "test_pipeline: ${_dll} not in ${_sycl_bin} (PATH at runtime)")
        endif()
    endforeach()
endif()

# ggml/whisper/ggml-sycl now ship as SHARED DLLs (see top-level CMakeLists.txt);
# co-locate them beside the test exe so it loads at runtime.
whisper_xpu_copy_shared_libs(test_pipeline)
