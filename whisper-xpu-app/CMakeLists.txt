add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
    src/sched_log.cpp
    src/transcription_scheduler.cpp
    zh_converter.cpp
)

# Vendored OpenCC text dicts (third_party/opencc-data) — read at runtime by
# zh_converter for the Simplified/Traditional Chinese-display modes.
target_compile_definitions(whisper_xpu_app PRIVATE
    WHISPER_XPU_OPENCC_DATA_DIR="${CMAKE_SOURCE_DIR}/third_party/opencc-data")

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)

target_compile_options(whisper_xpu_app PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

target_link_libraries(whisper_xpu_app PRIVATE
    whisper_xpu_sycl_core
    wx::core
    portaudio
)

if(WIN32)
    target_link_libraries(whisper_xpu_app PRIVATE
        winmm
        ole32
        uuid
        comctl32
        rpcrt4
    )
endif()

# ── Copy project DLL next to the executable ──
add_custom_command(TARGET whisper_xpu_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:whisper_xpu_app>
    COMMENT "Copying whisper_xpu_sycl_core.dll to app output directory"
)
# ggml/whisper/ggml-sycl now ship as SHARED DLLs (see top-level CMakeLists.txt);
# co-locate them beside the app exe so it loads at runtime.
whisper_xpu_copy_shared_libs(whisper_xpu_app)

# ── Copy Intel oneAPI runtime DLLs ──
# These are needed by whisper_xpu_sycl_core.dll + ggml-sycl.dll at runtime.
# oneAPI 2026.1: sycl8.dll → sycl9.dll; tcm.dll moved to tcm/latest/bin.
# Every copy is EXISTS-guarded so a relocated DLL never fails the build.
# ze_loader.dll is NOT copied — it's part of the Intel GPU driver (lives in
# System32) and must be present on the target system for GPU functionality.
# mkl_sycl_blas.5.dll + its MKL deps ARE copied: ggml-sycl.dll imports
# oneMKL SYCL BLAS (gemm/trsm) for its GEMM path; without these the app fails
# to load ggml-sycl.dll ("missing mkl_sycl_blas.5.dll") when launched without
# oneapi-vars on PATH (double-click / no env).  Found in <oneapi>/mkl/latest/bin
# (2025.3) — SYCL_LIBRARY_DIR is under compiler/, so the MKL bin is a sibling.
if(WIN32 AND SYCL_LIBRARY_DIR)
    # SYCL_LIBRARY_DIR = <oneapi>/<ver>/lib, so ../bin = <oneapi>/<ver>/bin,
    # which holds BOTH the SYCL runtime DLLs and the oneMKL runtime DLLs (they
    # are co-located in a oneAPI install).
    get_filename_component(_sycl_bin "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll
        ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll
        # oneMKL SYCL runtime + its CPU-side deps: ggml-sycl.dll imports
        # mkl_sycl_blas.5.dll (oneMKL SYCL BLAS — gemm/trsm), which in turn
        # needs mkl_core/mkl_intel_thread.  Without these the app fails to load
        # ggml-sycl.dll ("missing mkl_sycl_blas.5.dll") when launched without
        # oneapi-vars on PATH (double-click / no env).
        mkl_sycl_blas.5.dll mkl_sycl_lapack.5.dll mkl_sycl_rng.5.dll
        mkl_core.2.dll mkl_intel_thread.2.dll mkl_def.2.dll mkl_avx2.2.dll
        mkl_avx512.2.dll mkl_vml_def.2.dll mkl_vml_avx2.2.dll)
    foreach(_dll ${_rt_dlls})
        if(EXISTS "${_sycl_bin}/${_dll}")
            add_custom_command(TARGET whisper_xpu_app POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_sycl_bin}/${_dll}"
                    $<TARGET_FILE_DIR:whisper_xpu_app>
                COMMENT "Copying ${_dll} to app output directory")
        else()
            message(STATUS "whisper_xpu_app: ${_dll} not in ${_sycl_bin} (PATH at runtime)")
        endif()
    endforeach()
endif()
