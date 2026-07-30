add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
    src/sched_log.cpp
    src/transcription_scheduler.cpp
    zh_converter.cpp
    app.rc
)

# Vendored OpenCC text dicts (third_party/opencc-data) — read at runtime by
# zh_converter for the Simplified/Traditional Chinese-display modes.
# Release packaging resolves the dicts RELATIVE to the exe's dir (models/opencc,
# copied into the zip) so a drop-and-run install has no absolute source path;
# dev builds keep the absolute source path so dicts resolve from the build dir.
if(WHISPER_XPU_RELEASE)
    target_compile_definitions(whisper_xpu_app PRIVATE
        WHISPER_XPU_OPENCC_DATA_DIR="models/opencc")
else()
    target_compile_definitions(whisper_xpu_app PRIVATE
        WHISPER_XPU_OPENCC_DATA_DIR="${CMAKE_SOURCE_DIR}/third_party/opencc-data")
endif()

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
# ── Copy the app icon PNG (loaded at runtime by SetIcon for the title bar) ──
add_custom_command(TARGET whisper_xpu_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/app256.png"
        $<TARGET_FILE_DIR:whisper_xpu_app>
    COMMENT "Copying app icon to app output directory"
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

# ── Release packaging (WHISPER_XPU_RELEASE=ON) ──────────────────────────────
# Build a drop-and-run zip: unzip → double-click exe → works (no oneAPI env,
# no source tree).  cmake -B build-gpu -DWHISPER_XPU_RELEASE=ON, then
# cmake --build build-gpu --config Release --target release_package.  OFF by
# default (dev builds don't pay the packaging cost).
#
# Layout inside <build>/release/:
#   whisper_xpu_app.exe + ALL DLLs (ggml/whisper/sycl/oneAPI/MKL/L0)
#   app256.png (title-bar icon)
#   whisper_xpu.ini (defaults: tiny model, CPU device — safe, runs anywhere)
#   models/ggml-tiny.bin, models/vad_model/ggml-vad.bin
#   models/opencc/{config,dictionary,...}   (zh converter dicts, exe-relative)
# → zipped to whisper_xpu_v${WHISPER_XPU_VERSION}.zip beside release/
if(WHISPER_XPU_RELEASE)
    set(WHISPER_XPU_VERSION "${PROJECT_VERSION}")
    set(_rel_dir "${CMAKE_BINARY_DIR}/release")
    set(_exe "$<TARGET_FILE:whisper_xpu_app>")
    set(_app_dir "$<TARGET_FILE_DIR:whisper_xpu_app>")
    set(_zip "${CMAKE_BINARY_DIR}/whisper_xpu_v${WHISPER_XPU_VERSION}.zip")

    add_custom_target(release_package
        # 1. Fresh release/ dir
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_rel_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_rel_dir}"

        # 2. App exe + icon png + core DLL (whisper_xpu_sycl_core.dll)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_exe}"        "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/resources/app256.png" "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:whisper_xpu_sycl_core> "${_rel_dir}/"

        # 3. ggml/whisper shared DLLs (base/cpu/sycl/ggml/whisper) — same set
        #    the POST_BUILD step co-locates; copy from the built app dir (they
        #    are already there from the normal build's POST_BUILD copies).
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_app_dir}/ggml-base.dll"     "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_app_dir}/ggml-cpu.dll"      "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_app_dir}/ggml-sycl.dll"     "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_app_dir}/ggml.dll"          "${_rel_dir}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_app_dir}/whisper.dll"       "${_rel_dir}/"
        VERBATIM
        COMMENT "release_package: exe + ggml/whisper DLLs")

    # 3b. oneAPI runtime + MKL + Level Zero DLLs (same list as the dev
    #     POST_BUILD copy).  Re-derive _sycl_bin here (it's scoped to the
    #     earlier if-block).  Copy from the oneAPI install (EXISTS-guarded).
    if(WIN32 AND SYCL_LIBRARY_DIR)
        get_filename_component(_rel_sycl_bin "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
        set(_rel_rt_dlls sycl7.dll sycl8.dll sycl9.dll svml_dispmd.dll libmmd.dll
            libiomp5md.dll ur_win_proxy_loader.dll ur_loader.dll
            ur_adapter_level_zero.dll ur_adapter_opencl.dll
            umf.dll tcm.dll ur_adapter_level_zero_v2.dll libhwloc-15.dll sycl-jit.dll
            mkl_sycl_blas.5.dll mkl_sycl_lapack.5.dll mkl_sycl_rng.5.dll
            mkl_core.2.dll mkl_intel_thread.2.dll mkl_def.2.dll mkl_avx2.2.dll
            mkl_avx512.2.dll mkl_vml_def.2.dll mkl_vml_avx2.2.dll)
        foreach(_dll ${_rel_rt_dlls})
            if(EXISTS "${_rel_sycl_bin}/${_dll}")
                add_custom_command(TARGET release_package POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_rel_sycl_bin}/${_dll}" "${_rel_dir}/"
                    COMMENT "release_package: copy ${_dll}")
            else()
                message(STATUS "release_package: ${_dll} not found in ${_rel_sycl_bin} (skipped)")
            endif()
        endforeach()
    endif()

    # 4. Models: tiny (small, runs anywhere) + VAD model (in its own subdir,
    #    mirroring the source-tree layout models/vad_model/).
    add_custom_command(TARGET release_package POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_rel_dir}/models"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_rel_dir}/models/vad_model"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/ggml-tiny.bin" "${_rel_dir}/models/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/vad_model/ggml-vad.bin" "${_rel_dir}/models/vad_model/"
        COMMENT "release_package: models")

    # 5. OpenCC data → models/opencc (zh_converter reads WHISPER_XPU_OPENCC_DATA_DIR,
    #    which the release build defines as "models/opencc" — exe-relative).
    add_custom_command(TARGET release_package POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/third_party/opencc-data" "${_rel_dir}/models/opencc"
        COMMENT "release_package: opencc data → models/opencc")

    # 6. Default whisper_xpu.ini: tiny model + CPU device (safe everywhere —
    #    no GPU driver dependency on first run).  User edits via Settings.
    file(WRITE "${CMAKE_BINARY_DIR}/whisper_xpu.ini"
        "model=models/ggml-tiny.bin\n"
        "device=-1\n"
        "mic=-1\n"
        "hotkey=Ctrl+Shift+R\n"
        "zh=1\n")
    add_custom_command(TARGET release_package POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/whisper_xpu.ini" "${_rel_dir}/"
        COMMENT "release_package: default whisper_xpu.ini")

    # 7. Zip release/ → whisper_xpu_v<ver>.zip (PowerShell Compress-Archive).
    add_custom_command(TARGET release_package POST_BUILD
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command
            "Remove-Item -Force '${_zip}' -ErrorAction SilentlyContinue; Compress-Archive -Path '${_rel_dir}/*' -DestinationPath '${_zip}' -CompressionLevel Optimal"
        COMMAND ${CMAKE_COMMAND} -E echo "release_package: wrote ${_zip}"
        COMMENT "release_package: zip")
    add_dependencies(release_package whisper_xpu_app)
endif()
