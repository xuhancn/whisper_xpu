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

# ── App-sequence repro: same scheduler start path as the app (no
# transcribe_file warmup).  Build against the SAME Engine/scheduler sources.
# Mirrors test_pipeline's compile/link + DLL-copy setup.
add_executable(test_app_seq
    test_app_seq.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/transcription_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/sched_log.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/audio_capture.cpp
)
target_include_directories(test_app_seq PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)
target_compile_options(test_app_seq PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_compile_definitions(test_app_seq PRIVATE
    TEST_WAV_DIR="${CMAKE_SOURCE_DIR}/tests/bench_vad"
)
target_link_libraries(test_app_seq PRIVATE
    whisper_xpu_sycl_core
    portaudio
)
if(WIN32)
    target_link_libraries(test_app_seq PRIVATE winmm ole32 uuid)
endif()

# Co-locate the SYCL core DLL + oneAPI runtime + ggml/whisper shared libs next
# to test_app_seq, identical to test_pipeline's POST_BUILD copies.
add_custom_command(TARGET test_app_seq POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:test_app_seq>
    COMMENT "Copying whisper_xpu_sycl_core.dll to test_app_seq output directory"
)
if(WIN32 AND SYCL_LIBRARY_DIR)
    get_filename_component(_sycl_bin2 "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls2 sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll
        ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll)
    foreach(_dll ${_rt_dlls2})
        if(EXISTS "${_sycl_bin2}/${_dll}")
            add_custom_command(TARGET test_app_seq POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_sycl_bin2}/${_dll}"
                    $<TARGET_FILE_DIR:test_app_seq>
                COMMENT "Copying ${_dll} to test_app_seq output directory")
        endif()
    endforeach()
endif()
whisper_xpu_copy_shared_libs(test_app_seq)

# ── Robustness tests: rapid model switch, record-before-load, reload-while-
# recording.  Validates the crash guards (bounded join, whole-sequence lock,
# null-engine launch gate, clean stop-while-recording).
add_executable(test_robust
    test_robust.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/transcription_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/sched_log.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/audio_capture.cpp
)
target_include_directories(test_robust PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)
target_compile_options(test_robust PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_compile_definitions(test_robust PRIVATE
    TEST_WAV_DIR="${CMAKE_SOURCE_DIR}/tests/bench_vad"
)
target_link_libraries(test_robust PRIVATE
    whisper_xpu_sycl_core
    portaudio
)
if(WIN32)
    target_link_libraries(test_robust PRIVATE winmm ole32 uuid)
endif()
add_custom_command(TARGET test_robust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:test_robust>
    COMMENT "Copying whisper_xpu_sycl_core.dll to test_robust output directory"
)
if(WIN32 AND SYCL_LIBRARY_DIR)
    get_filename_component(_sycl_bin4 "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls4 sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll
        ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll)
    foreach(_dll ${_rt_dlls4})
        if(EXISTS "${_sycl_bin4}/${_dll}")
            add_custom_command(TARGET test_robust POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_sycl_bin4}/${_dll}"
                    $<TARGET_FILE_DIR:test_robust>
                COMMENT "Copying ${_dll} to test_robust output directory")
        endif()
    endforeach()
endif()
whisper_xpu_copy_shared_libs(test_robust)

target_compile_options(test_pipeline PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

target_compile_definitions(test_pipeline PRIVATE
    TEST_WAV_DIR="${CMAKE_SOURCE_DIR}/tests/bench_vad"
)

# ── Warmup variant of test_app_seq: adds a transcribe_file warmup before the
# scheduler starts, to confirm the warmup is what makes test_pipeline pass.
add_executable(test_app_seq_warm
    test_app_seq_warm.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/transcription_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/src/sched_log.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/audio_capture.cpp
)
target_include_directories(test_app_seq_warm PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)
target_compile_options(test_app_seq_warm PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_compile_definitions(test_app_seq_warm PRIVATE
    TEST_WAV_DIR="${CMAKE_SOURCE_DIR}/tests/bench_vad"
)
target_link_libraries(test_app_seq_warm PRIVATE
    whisper_xpu_sycl_core
    portaudio
)
if(WIN32)
    target_link_libraries(test_app_seq_warm PRIVATE winmm ole32 uuid)
endif()
add_custom_command(TARGET test_app_seq_warm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:test_app_seq_warm>
    COMMENT "Copying whisper_xpu_sycl_core.dll to test_app_seq_warm output directory"
)
if(WIN32 AND SYCL_LIBRARY_DIR)
    get_filename_component(_sycl_bin3 "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls3 sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll
        ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll)
    foreach(_dll ${_rt_dlls3})
        if(EXISTS "${_sycl_bin3}/${_dll}")
            add_custom_command(TARGET test_app_seq_warm POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_sycl_bin3}/${_dll}"
                    $<TARGET_FILE_DIR:test_app_seq_warm>
                COMMENT "Copying ${_dll} to test_app_seq_warm output directory")
        endif()
    endforeach()
endif()
whisper_xpu_copy_shared_libs(test_app_seq_warm)

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
